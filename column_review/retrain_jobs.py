"""Subprocess wrapper for `scripts/retrain_yolo.py` background jobs.

`start_retrain(...)` spawns the retrain CLI as a `subprocess.Popen`,
inserts a row into the `retrain_jobs` table (`queued` initially), and
returns the new row's id. A background daemon thread polls live Popens
every 2 seconds and flips the status to `running`/`completed`/`failed`
as the process progresses.

The subprocess survives the column-review server lifetime — that is by
design: a retrain takes minutes, the reviewer may close the browser tab
mid-job, and we don't want to kill the GPU work. The trade-off is that
on a server restart, the database may carry a `running` row whose PID
is long dead. `reap_orphans()` runs once at startup and marks any
such row as `failed: orphaned (server restarted)` so the UI doesn't
report a phantom job forever.

Concurrency: writes to `retrain_jobs` use a fresh `sqlite3.connect` per
update; live Popen objects live in a process-local dict guarded by a
threading.Lock.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from column_review.db import get_connection


# Live Popen objects keyed by the `retrain_jobs.id` they correspond to.
# Cleared when the poller observes a terminal status. Module-level so
# the poller daemon can read it; lock because both `start_retrain` and
# the poller mutate the dict.
_LIVE_PROCS: dict[int, subprocess.Popen] = {}
_LIVE_PROCS_LOCK = threading.Lock()

# Poll interval. Tuned to feel responsive in the UI (~2 s for the
# status-pill flip) without busy-waiting on retrain runs that take
# minutes-to-hours.
_POLL_INTERVAL_S = 2.0

# Stderr tail size kept in the database for failure surfacing. Enough
# for an ultralytics traceback + the last few log lines without
# bloating the row.
_STDERR_TAIL_BYTES = 64 * 1024

# Set True by `start_poller_thread()` so the daemon thread is only
# launched once per process even if `create_app` is called multiple
# times in tests.
_POLLER_STARTED = False
_POLLER_LOCK = threading.Lock()

# Per-job-id tee threads — read stdout/stderr from the subprocess
# line-by-line and append to `data/jobs/retrain/<job_id>.log`.
# Keyed by retrain_jobs.id so the poller can clean up references.
_TEE_THREADS: dict[int, threading.Thread] = {}
_LOGS_DIR: Path = None  # set on first start_retrain — relative to project_root


def _logs_dir(project_root: Path) -> Path:
    """Resolve and create `<project>/data/jobs/retrain/` for log tees."""
    p = project_root / "data" / "jobs" / "retrain"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tee_stream(stream, log_path: Path, label: str) -> None:
    """Read `stream` line-by-line and append each line to `log_path`.

    Also echoes to the server's stdout so the user's `column-review`
    terminal shows the retrain progress live. Runs on a daemon
    thread; exits when the stream EOFs (subprocess closed it).
    """
    try:
        with open(log_path, "a", encoding="utf-8", buffering=1) as f:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                f.write(line)
                # Also echo to server stdout with a [retrain] tag so the
                # user can watch the retrain in their terminal too.
                sys.stdout.write(f"[retrain] {line}")
                sys.stdout.flush()
    except Exception as e:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[tee-{label}-error] {e}\n")
        except OSError:
            pass


def log_path_for(retrain_job_id: int, project_root: Path) -> Path:
    """Resolve the on-disk log file for a `retrain_jobs.id`."""
    return _logs_dir(project_root) / f"{retrain_job_id}.log"


def log_tail(retrain_job_id: int, project_root: Path,
             n_lines: int = 200) -> str:
    """Read the last `n_lines` of a retrain job's log."""
    p = log_path_for(retrain_job_id, project_root)
    if not p.is_file():
        return ""
    try:
        with open(p, "rb") as f:
            try:
                # Seek backwards from end up to ~256 KB to grab the tail
                # cheaply without loading the whole file.
                f.seek(0, 2)
                size = f.tell()
                window = min(size, 256 * 1024)
                f.seek(size - window)
                tail = f.read().decode("utf-8", errors="replace")
            except OSError:
                f.seek(0)
                tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = tail.splitlines()
    return "\n".join(lines[-n_lines:])


def start_retrain(epochs: int, min_corrections: int,
                  project_root: Path,
                  db_path: Optional[Path] = None) -> dict:
    """Spawn `scripts/retrain_yolo.py` as a background subprocess.

    Returns `{job_id, pid, started_ts}`. stdout + stderr are tee'd
    to `data/jobs/retrain/<retrain_job_id>.log` AND echoed to the
    server's stdout (the user's `column-review` terminal) so the
    user can monitor progress live in both places.
    """
    cmd = [
        sys.executable, str(project_root / "scripts" / "retrain_yolo.py"),
        "--epochs", str(epochs),
        "--min-corrections", str(min_corrections),
    ]
    # Pipe so we can tee. `bufsize=1, text=True` forces line buffering
    # so the tee thread sees progress lines as they arrive instead of
    # waiting for a 4 KB block.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        cwd=str(project_root),
        bufsize=1,
        text=True,
    )
    started_ts = time.time()
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO retrain_jobs "
            "(pid, started_ts, status, stderr_tail) "
            "VALUES (?, ?, ?, ?)",
            (proc.pid, started_ts, "queued", None),
        )
        job_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    # Spawn the tee thread BEFORE registering the proc so the poller
    # doesn't see an exit-without-output race.
    log_path = log_path_for(job_id, project_root)
    log_path.write_text(
        f"[retrain] spawned pid={proc.pid} epochs={epochs} "
        f"min_corrections={min_corrections} at {started_ts}\n",
        encoding="utf-8",
    )
    t = threading.Thread(
        target=_tee_stream,
        args=(proc.stdout, log_path, "out"),
        daemon=True,
        name=f"retrain-tee-{job_id}",
    )
    t.start()
    _TEE_THREADS[job_id] = t
    with _LIVE_PROCS_LOCK:
        _LIVE_PROCS[job_id] = proc
    print(
        f"[retrain] spawned pid={proc.pid} job_id={job_id} "
        f"epochs={epochs} min_corrections={min_corrections} "
        f"log={log_path}",
        flush=True,
    )
    return {"job_id": job_id, "pid": proc.pid, "started_ts": started_ts,
            "log_path": str(log_path)}


def _read_stderr_tail(proc: subprocess.Popen) -> str:
    """Read up to `_STDERR_TAIL_BYTES` of the proc's stdout (we
    redirected stderr→stdout). Tee thread has been writing the same
    data to disk in parallel; this is a defensive backstop for the
    poller's status update."""
    if proc.stdout is None:
        return ""
    try:
        # In text mode, .read() returns str; convert to bytes for
        # tail trim.
        data = proc.stdout.read() or ""
    except Exception:
        return ""
    if isinstance(data, str):
        data = data.encode("utf-8", errors="replace")
    if len(data) > _STDERR_TAIL_BYTES:
        data = data[-_STDERR_TAIL_BYTES:]
    return data.decode("utf-8", errors="replace")


def _poll_loop(db_path: Optional[Path]) -> None:
    """Daemon thread: poll every live Popen, flip DB statuses on exit.

    On each tick, snapshots the live-procs dict, calls `.poll()` on
    each, and updates the DB for any that completed. Status flips:
      queued/running → running (if .poll() returns None)
      running → completed (if exit code == 0)
      running → failed   (if exit code != 0)
    """
    while True:
        time.sleep(_POLL_INTERVAL_S)
        with _LIVE_PROCS_LOCK:
            jobs = list(_LIVE_PROCS.items())
        if not jobs:
            continue
        for job_id, proc in jobs:
            rc = proc.poll()
            if rc is None:
                # Still running — flip queued → running on first sight.
                conn = get_connection(db_path)
                try:
                    conn.execute(
                        "UPDATE retrain_jobs SET status = 'running' "
                        "WHERE id = ? AND status = 'queued'",
                        (job_id,),
                    )
                    conn.commit()
                finally:
                    conn.close()
                continue
            # Terminal — capture stderr tail and update DB.
            status = "completed" if rc == 0 else "failed"
            stderr_tail = _read_stderr_tail(proc)
            conn = get_connection(db_path)
            try:
                conn.execute(
                    "UPDATE retrain_jobs SET status = ?, "
                    "finished_ts = ?, stderr_tail = ? WHERE id = ?",
                    (status, time.time(), stderr_tail, job_id),
                )
                conn.commit()
            finally:
                conn.close()
            with _LIVE_PROCS_LOCK:
                _LIVE_PROCS.pop(job_id, None)
            print(
                f"[retrain] job_id={job_id} pid={proc.pid} "
                f"exit={rc} → {status}",
                flush=True,
            )


def start_poller_thread(db_path: Optional[Path]) -> None:
    """Launch the daemon poller exactly once per process."""
    global _POLLER_STARTED
    with _POLLER_LOCK:
        if _POLLER_STARTED:
            return
        threading.Thread(
            target=_poll_loop,
            args=(db_path,),
            daemon=True,
            name="column-review-retrain-poller",
        ).start()
        _POLLER_STARTED = True


def reap_orphans(db_path: Optional[Path]) -> int:
    """Mark `queued`/`running` rows whose PID is dead as `failed`.

    Returns the number of rows updated. Called once at server startup
    so the UI doesn't report a phantom retrain forever after a crash
    or restart that killed only this process and not the spawned
    subprocess. (If the subprocess was killed too, the row was already
    invalid; if the subprocess survived, we'll re-discover it on its
    own merits via the live-procs dict next time it's spawned.)
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, pid FROM retrain_jobs "
            "WHERE status IN ('queued', 'running')"
        ).fetchall()
        n_reaped = 0
        for job_id, pid in rows:
            if pid is None:
                continue
            try:
                # `os.kill(pid, 0)` raises ProcessLookupError if the
                # PID does not exist. (Permission errors mean the PID
                # IS alive but owned by another user — leave alone.)
                os.kill(pid, 0)
            except ProcessLookupError:
                conn.execute(
                    "UPDATE retrain_jobs SET status = 'failed', "
                    "finished_ts = ?, stderr_tail = ? WHERE id = ?",
                    (time.time(),
                     "orphaned (server restarted while job was running)",
                     job_id),
                )
                n_reaped += 1
            except PermissionError:
                # Different-user-owned PID — leave the row alone.
                pass
        if n_reaped:
            conn.commit()
            print(f"[retrain] reaped {n_reaped} orphan job(s)", flush=True)
        return n_reaped
    finally:
        conn.close()


def latest_job(db_path: Optional[Path]) -> Optional[dict]:
    """Return the most-recent `retrain_jobs` row as a dict, or None."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, pid, started_ts, status, finished_ts, "
            "       stderr_tail "
            "FROM retrain_jobs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {
        "id":          row[0],
        "pid":         row[1],
        "started_ts":  row[2],
        "status":      row[3],
        "finished_ts": row[4],
        "stderr_tail": row[5],
    }


def corrections_count(job_id: str,
                      db_path: Optional[Path]) -> dict:
    """Return `{n_fp, n_fn_added, n_total_corrections}` for the job.

    Drives the confirm-dialog preview text. Counts are *effective* —
    rescinded deletes are filtered by the same `iter_effective_corrections`
    helper retrain consumes, so the user sees the same numbers retrain
    will see.
    """
    from column_review.db import iter_effective_corrections
    conn = get_connection(db_path)
    try:
        n_fp = n_fn = n_total = 0
        for row in iter_effective_corrections(conn, job_id=job_id):
            _job, _et, _idx, _orig, _changes, is_delete, _ts = row
            n_total += 1
            if is_delete:
                n_fp += 1
            else:
                n_fn += 1
    finally:
        conn.close()
    return {"n_fp": n_fp, "n_fn_added": n_fn, "n_total": n_total}
