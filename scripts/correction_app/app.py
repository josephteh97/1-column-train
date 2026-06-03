"""FastAPI backend for the column-detection correction reviewer.

Single drawing per process. The drawing is supplied to `create_app` and
locked in for the life of the server. The app exposes:

- `GET /`                          → serves static/index.html
- `GET /static/*`                  → served from scripts/correction_app/static/
- `GET /dzi/<id>.dzi`              → DZI manifest under data/raw/drawings/
- `GET /dzi/<id>_files/...`        → DZI tile JPEGs
- `GET /api/drawing`               → drawing + job + detections + state + config
- `GET /api/state`                 → consolidated four-state map per element_index
- `POST /api/marks`                → single mark write (TP/FP/FN_ADDED/RESCIND_FP/DELETE_FN)
- `POST /api/marks/batch`          → batch mark writes in one transaction
- `HEAD /api/dzi-exists`           → 200 if DZI present, 404 otherwise
- `POST /api/session`              → set/replace reviewer_id, insert reviewer_sessions row

The corrections-DB write contract is satisfied entirely via
`scripts.corrections_logger`: this module never writes directly to the
existing `corrections` table. The two new sidecar tables
(`tp_confirmations`, `reviewer_sessions`) are created here by
idempotent `CREATE TABLE IF NOT EXISTS` at app start.
"""
from __future__ import annotations

import json
import socket
import sqlite3
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Optional

# Path discipline: this module lives at scripts/correction_app/app.py.
# Anchor all paths to the project root via __file__ so subprocess /
# uvicorn-launched callers see the same files as `scripts/hitl.py`.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCRIPTS_DIR = HERE.parent

# Allow `from corrections_logger import ...` and `from ingest_drawings
# import ...` without the user installing this package.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from corrections_logger import (    # noqa: E402  — sys.path tweak above
    new_job_id,
    iter_effective_corrections,
    DB_PATH,
    JOBS_DIR,
    DATA_ROOT,
)
# `_apply_marks` is the single writer into `data/corrections.db`; see
# openspec/changes/rebuild-correction-ui-web/design.md for rationale.
from ingest_drawings import resolve_drawing  # noqa: E402

# FastAPI is an optional dep — checked at hitl.py review entry, so we
# can import unconditionally here once we know hitl already gate-kept.
from fastapi import FastAPI, HTTPException, Request   # noqa: E402
from fastapi.responses import (                       # noqa: E402
    FileResponse,
    JSONResponse,
    HTMLResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles           # noqa: E402

STATIC_DIR = HERE / "static"
RAW_DRAWINGS_DIR = DATA_ROOT / "raw" / "drawings"

REVIEWER_CONFIG_PATH = Path.home() / ".column-review.json"

# Sidecar-table DDL — strictly additive, no ALTER on existing tables.
SIDECAR_DDL = """
CREATE TABLE IF NOT EXISTS tp_confirmations (
    session_id    TEXT,
    job_id        TEXT,
    element_index INTEGER,
    ts            REAL,
    PRIMARY KEY (job_id, element_index)
);
CREATE TABLE IF NOT EXISTS reviewer_sessions (
    session_id  TEXT PRIMARY KEY,
    reviewer_id TEXT NOT NULL,
    started_ts  REAL NOT NULL
);
"""


def _ensure_sidecar_tables() -> None:
    """Idempotent additive migration. CREATE TABLE IF NOT EXISTS is a
    no-op when the tables already exist, and `data/corrections.db`'s
    existing `corrections` table is not touched at all.

    Also delegates to `corrections_logger._ensure_db()` so the main
    `corrections` table is guaranteed to exist before `_apply_marks`
    issues its inline SQL. Without this, a fresh install where no
    `record_*` call has ever run would fail with `no such table:
    corrections` on the first POST /api/marks.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    # corrections_logger._ensure_db returns an open connection that runs
    # the canonical `corrections` table DDL. We close it immediately —
    # _apply_marks opens its own conn per request.
    import corrections_logger as _cl
    _cl._ensure_db().close()
    # Then layer the additive sidecar tables on top.
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.executescript(SIDECAR_DDL)
        conn.commit()
    finally:
        conn.close()


def _load_reviewer_id() -> Optional[str]:
    if not REVIEWER_CONFIG_PATH.exists():
        return None
    try:
        return json.loads(REVIEWER_CONFIG_PATH.read_text()).get("reviewer_id")
    except (json.JSONDecodeError, OSError):
        return None


def _save_reviewer_id(reviewer_id: str) -> None:
    """Atomic-replace write. A crash mid-write must not leave a
    truncated JSON file that the next launch's `_load_reviewer_id`
    silently treats as 'no reviewer-id' (causing the prompt bar to
    reappear and orphaning reviewer_sessions rows already inserted
    under the now-lost id).
    """
    import os
    tmp = REVIEWER_CONFIG_PATH.with_suffix(REVIEWER_CONFIG_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps({"reviewer_id": reviewer_id}))
    os.replace(tmp, REVIEWER_CONFIG_PATH)


def _start_session(reviewer_id: str) -> str:
    """Insert a fresh `reviewer_sessions` row, return the new session_id."""
    session_id = uuid.uuid4().hex
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            "INSERT INTO reviewer_sessions (session_id, reviewer_id, started_ts) "
            "VALUES (?, ?, ?)",
            (session_id, reviewer_id, time.time()),
        )
        conn.commit()
    finally:
        conn.close()
    return session_id


def _bootstrap_empty_job(job_id: str, source_path: str,
                         raster_mtime: float) -> None:
    """Write the minimal job directory for a fresh review session.

    Critically does NOT encode render.jpg here — that is offloaded to
    `_spawn_render_jpg_write` because JPEG-encoding the full A0/300DPI
    raster (~140 Mpx) takes 10-30 s and would block create_app, busting
    the spec's <3 s open requirement. The reviewer never reads
    render.jpg; it is consumed downstream by hard_negative_pool at
    retrain time, by which point the background encode has long
    completed.
    """
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    detections = {
        "columns": [],
        "meta": {
            "source":       source_path,
            "created_ts":   time.time(),
            "raster_mtime": raster_mtime,
            "n":            0,
        },
    }
    (job_dir / "px_detections.json").write_text(json.dumps(detections, indent=2))


def _spawn_render_jpg_write(job_id: str, raster_path: Path) -> None:
    """Background-thread render.jpg writer. Idempotent — skips if the
    file already exists. Drops `optimize=True` (which can multiply
    encode time on 100+ Mpx images) since render.jpg is consumed by
    hard_negative_pool for cropping, not for redistribution: bytes-on-
    disk size is not a constraint here, latency is.
    """
    def _do():
        render_path = JOBS_DIR / job_id / "render.jpg"
        if render_path.exists():
            return
        try:
            from PIL import Image
            with Image.open(raster_path) as src:
                src.convert("RGB").save(render_path, quality=92)
        except Exception:   # noqa: BLE001 — log + drop on the background thread
            import traceback
            traceback.print_exc()
    threading.Thread(target=_do, daemon=True).start()


def _find_or_create_job(drawing_id: str, raster_path: Path,
                       source_path: str) -> str:
    """Reuse an existing job for this drawing if its `meta.source` AND
    `meta.raster_mtime` BOTH match; otherwise create a fresh job. The
    raster_mtime guard ensures a re-ingested drawing (same source path
    on disk, but freshly written pixels) doesn't inherit the stale
    render.jpg of the previous job — which would silently misalign
    hard_negative_pool's crops with the new bbox coordinates.
    """
    raster_mtime = raster_path.stat().st_mtime
    if JOBS_DIR.exists():
        # Newest-first: most recently mutated job wins among matches.
        candidates = sorted(JOBS_DIR.glob("*/px_detections.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            try:
                det = json.loads(cand.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            meta = det.get("meta", {})
            if meta.get("source") != source_path:
                continue
            stored_mtime = meta.get("raster_mtime")
            if stored_mtime is None:
                # Legacy job created before raster_mtime was recorded.
                # Match by source alone (preserves the old behaviour
                # for jobs that already hold real corrections); tag
                # the meta so we get strict matching on the next launch.
                meta["raster_mtime"] = raster_mtime
                det["meta"] = meta
                # Atomic write — plain write_text would leave a truncated
                # px_detections.json on a crash and dead-lock the session.
                import os
                tmp = cand.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(det, indent=2))
                os.replace(tmp, cand)
                return cand.parent.name
            if stored_mtime == raster_mtime:
                return cand.parent.name
            # mtime mismatch → fall through and bootstrap a fresh job.
            # The stale job is left in place so its correction history
            # remains intact for any prior retrain run.

    job_id = new_job_id()
    _bootstrap_empty_job(job_id, source_path, raster_mtime)
    _spawn_render_jpg_write(job_id, raster_path)
    return job_id


def _state_map_from(cols: list, job_id: str,
                    conn: Optional[sqlite3.Connection] = None) -> dict:
    """Pure compute-the-state-map step. Takes the already-loaded `cols`
    list and either an existing connection (re-used inside a transaction
    by `_apply_marks`) or opens its own. Splitting this out of
    `_build_state_map` lets the mark-apply path avoid a second read of
    `px_detections.json` per request — that was the dominant cost on
    high-rate sessions and is the fix for code-review #6.
    """
    state: dict[str, str] = {}
    for i, c in enumerate(cols):
        state[str(i)] = ("FN_ADDED" if c.get("source") == "human_added"
                         else "UNREVIEWED")
    owns_conn = conn is None
    if owns_conn:
        conn = sqlite3.connect(str(DB_PATH))
    try:
        for row in iter_effective_corrections(conn, job_id=job_id):
            _job, _et, idx, _orig, changes_json, is_del, _ts = row
            if not is_del:
                continue
            # DELETE_FN tags `changes={"action":"delete_fn"}` to mark the
            # row as "user undid their own add". Plain FP (including FP
            # cast on a human_added entry) leaves changes={} and stays
            # visible as FP. Without this marker we couldn't tell the
            # two apart — both are is_delete=1 on the same slot.
            try:
                changes = json.loads(changes_json) if changes_json else {}
            except (ValueError, TypeError):
                changes = {}
            if changes.get("action") == "delete_fn":
                state[str(idx)] = "REMOVED"
            else:
                state[str(idx)] = "FP"
        for (idx,) in conn.execute(
            "SELECT element_index FROM tp_confirmations WHERE job_id = ?",
            (job_id,),
        ):
            # TP cannot override an FP (the FP row is the audit trail)
            # nor a REMOVED entry (the slot's been retired).
            if state.get(str(idx)) not in ("FP", "REMOVED"):
                state[str(idx)] = "TP"
    finally:
        if owns_conn:
            conn.close()
    return state


def _build_state_map(job_id: str) -> dict:
    """File-reading wrapper around `_state_map_from`. Used by GET
    /api/state on page reload; the hot mark-apply path uses
    `_state_map_from` directly with the in-memory cols.
    """
    px_path = JOBS_DIR / job_id / "px_detections.json"
    if not px_path.exists():
        return {}
    det = json.loads(px_path.read_text())
    return _state_map_from(det.get("columns", []), job_id)


# ──────────────────────────────────────────────────────────────────────
# Single-transaction batch mark writer.
#
# Replaces the previous per-kind helpers (_record_tp / _remove_tp /
# _rescind_fp / _delete_fn) and the per-mark _apply_one wrapper. The
# whole batch — single mark OR rubber-band 500 — commits as ONE SQLite
# transaction, satisfying the spec's "one transaction" guarantee.
#
# The SQL here duplicates the INSERT statements that corrections_logger's
# record_* helpers emit. That duplication is the price of batching: the
# helpers open/commit/close their own connections, so threading a shared
# transaction through them would require modifying corrections_logger,
# which the OpenSpec design pinned as "preserved verbatim".
# ──────────────────────────────────────────────────────────────────────

_MARK_KINDS = {"TP", "CLEAR_TP", "FP", "FN_ADDED", "RESCIND_FP",
               "DELETE_FN", "RESTORE_FN"}


def _apply_marks(marks: list[dict], job_id: str,
                 session_id_val: str) -> dict:
    """Apply a list of marks in ONE SQLite transaction. Returns the
    fresh state map computed from the post-apply in-memory cols.

    Ordering for crash-safety: the JSON file is written FIRST via
    `os.replace` (atomic), then the DB transaction commits. If the
    process dies between the JSON write and the DB commit, the next
    launch sees the new JSON entry as a regular `human_added` column
    (no corrections row needed for it to count as a positive label
    downstream). The previous ordering (DB-commit-before-JSON-write)
    could leave the DB pointing at an element_index that JSON didn't
    have — silently losing the user's FN add.
    """
    if not marks:
        # Caller still expects a state map back. Cheap path.
        return _build_state_map(job_id)

    px_path = JOBS_DIR / job_id / "px_detections.json"
    if not px_path.exists():
        raise HTTPException(status_code=404, detail="px_detections.json missing")
    det = json.loads(px_path.read_text())
    cols = det.get("columns", [])
    json_dirty = False

    # Validate up-front so a malformed mark in the middle of a batch
    # doesn't leave a partial transaction behind.
    for m in marks:
        kind = m.get("kind")
        if kind not in _MARK_KINDS:
            raise HTTPException(status_code=400,
                                detail=f"unknown mark kind: {kind!r}")
        if kind != "FN_ADDED":
            try:
                idx = int(m["element_index"])
            except (KeyError, ValueError, TypeError):
                raise HTTPException(status_code=400,
                                    detail="element_index is required")
            # Bound-check uses the CURRENT cols length. FN_ADDED marks
            # in this batch would extend it, but they go through a
            # separate code path that doesn't bound-check.
            if not (0 <= idx < len(cols)):
                raise HTTPException(status_code=400,
                                    detail=f"element_index {idx} out of range")

    now = time.time()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Slots whose human_added entry has been DELETE_FN'd — used by
        # the FN_ADDED dedup so re-adding at the same rounded centre as
        # a previously-removed slot does NOT silently no-op against the
        # stale JSON entry.
        removed_human_idx: set[int] = set()
        for (rmidx,) in conn.execute(
            "SELECT element_index FROM corrections "
            "WHERE job_id = ? AND is_delete = 1",
            (job_id,),
        ).fetchall():
            if 0 <= rmidx < len(cols) and cols[rmidx].get("source") == "human_added":
                removed_human_idx.add(rmidx)

        for m in marks:
            kind = m["kind"]

            if kind == "TP":
                idx = int(m["element_index"])
                conn.execute(
                    "INSERT OR REPLACE INTO tp_confirmations "
                    "(session_id, job_id, element_index, ts) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id_val, job_id, idx, now),
                )

            elif kind == "CLEAR_TP":
                idx = int(m["element_index"])
                conn.execute(
                    "DELETE FROM tp_confirmations "
                    "WHERE job_id = ? AND element_index = ?",
                    (job_id, idx),
                )

            elif kind == "FP":
                idx = int(m["element_index"])
                # FP overrides any prior TP for the same detection.
                conn.execute(
                    "DELETE FROM tp_confirmations "
                    "WHERE job_id = ? AND element_index = ?",
                    (job_id, idx),
                )
                # If the slot is a previously-added FN, the FN_ADDED's
                # is_delete=0 row would silently RESCIND this FP at
                # every read site. Strip that add row first.
                if cols[idx].get("source") == "human_added":
                    conn.execute(
                        "DELETE FROM corrections "
                        "WHERE job_id = ? AND element_index = ? AND is_delete = 0",
                        (job_id, idx),
                    )
                conn.execute(
                    "INSERT OR IGNORE INTO corrections "
                    "(job_id, element_type, element_index, "
                    " original_element, changes, is_delete) "
                    "VALUES (?, 'column', ?, ?, '{}', 1)",
                    (job_id, idx, json.dumps(cols[idx])),
                )

            elif kind == "RESCIND_FP":
                idx = int(m["element_index"])
                # Mirror record_edit's sticky-original: preserve the
                # earlier is_delete=0 row's original_element if present,
                # otherwise capture from the current JSON.
                row = conn.execute(
                    "SELECT original_element FROM corrections "
                    "WHERE job_id = ? AND element_index = ? AND is_delete = 0",
                    (job_id, idx),
                ).fetchone()
                original_json = row[0] if row else json.dumps(dict(cols[idx]))
                bbox = cols[idx].get("bbox") or [0.0, 0.0, 0.0, 0.0]
                conn.execute(
                    "INSERT OR REPLACE INTO corrections "
                    "(job_id, element_type, element_index, "
                    " original_element, changes, is_delete) "
                    "VALUES (?, 'column', ?, ?, ?, 0)",
                    (job_id, idx, original_json,
                     json.dumps({"bbox": [float(x) for x in bbox]})),
                )

            elif kind == "DELETE_FN":
                idx = int(m["element_index"])
                # Bug-fix vs the prior implementation: deleting an
                # FN_ADDED's record_delete row alone gets RESCINDED by
                # the FN_ADDED's existing is_delete=0 row at every
                # read site. Drop the is_delete=0 row first so the new
                # is_delete=1 row stands alone and downstream consumers
                # (retrain_yolo, hard_negative_pool) see the deletion.
                #
                # Also tag changes={"action":"delete_fn"} so
                # _build_state_map can distinguish "user undid their
                # own add" (REMOVED → hidden in UI) from "user marked
                # the human_added entry as FP" (visible with FP styling).
                original = cols[idx]
                conn.execute(
                    "DELETE FROM corrections "
                    "WHERE job_id = ? AND element_index = ? AND is_delete = 0",
                    (job_id, idx),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO corrections "
                    "(job_id, element_type, element_index, "
                    " original_element, changes, is_delete) "
                    "VALUES (?, 'column', ?, ?, ?, 1)",
                    (job_id, idx, json.dumps(original),
                     json.dumps({"action": "delete_fn"})),
                )

            elif kind == "FN_ADDED":
                bbox = [float(x) for x in m["bbox"]]
                if len(bbox) < 4:
                    raise HTTPException(status_code=400,
                                        detail="bbox must have 4 values")
                # Dedup at the same rounded centre, mirroring record_add.
                # Skip slots already removed via DELETE_FN — those JSON
                # entries are no longer "live" from the UI's view.
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                rcx, rcy = round(cx), round(cy)
                deduped = False
                for ei, existing in enumerate(cols):
                    if existing.get("source") != "human_added":
                        continue
                    if ei in removed_human_idx:
                        continue
                    eb = existing.get("bbox") or []
                    if len(eb) < 4:
                        continue
                    if (round((eb[0] + eb[2]) / 2.0) == rcx
                            and round((eb[1] + eb[3]) / 2.0) == rcy):
                        deduped = True
                        break
                if deduped:
                    continue
                new_entry = {"bbox": bbox, "score": 1.0,
                             "source": "human_added"}
                cols.append(new_entry)
                new_idx = len(cols) - 1
                json_dirty = True
                conn.execute(
                    "INSERT OR IGNORE INTO corrections "
                    "(job_id, element_type, element_index, "
                    " original_element, changes, is_delete) "
                    "VALUES (?, 'column', ?, '{}', ?, 0)",
                    (job_id, new_idx,
                     json.dumps({"bbox": bbox, "source": "human_added"})),
                )

            elif kind == "RESTORE_FN":
                # Undo of DELETE_FN. The cols entry still has
                # source=human_added (DELETE_FN only mutates the DB,
                # not the JSON), so dropping the is_delete=1 audit row
                # and re-inserting the original is_delete=0 row
                # brings the slot back to FN_ADDED in _state_map_from.
                idx = int(m["element_index"])
                if cols[idx].get("source") != "human_added":
                    raise HTTPException(
                        status_code=400,
                        detail="RESTORE_FN target is not a human-added slot",
                    )
                conn.execute(
                    "DELETE FROM corrections "
                    "WHERE job_id = ? AND element_index = ? AND is_delete = 1",
                    (job_id, idx),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO corrections "
                    "(job_id, element_type, element_index, "
                    " original_element, changes, is_delete) "
                    "VALUES (?, 'column', ?, '{}', ?, 0)",
                    (job_id, idx,
                     json.dumps({"bbox": cols[idx].get("bbox"),
                                 "source": "human_added"})),
                )
                # Slot is no longer in the removed set; keep our local
                # cache in sync so further marks in the same batch see
                # the up-to-date state.
                removed_human_idx.discard(idx)

        # ── Crash-safety ordering: JSON FIRST, then DB commit. ──
        # If the JSON write fails (disk full, permission), the DB
        # transaction is rolled back via the `finally` close-without-
        # commit, so the cols length and corrections rows stay in
        # sync. If the DB commit fails after a successful JSON write,
        # the new entries appear on next launch as plain `human_added`
        # columns (no corrections row), which downstream still treats
        # as positive labels — the user's FN add is never silently
        # lost.
        if json_dirty:
            import os
            tmp = px_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(det, indent=2))
            os.replace(tmp, px_path)

        conn.commit()
        # Compute the state map BEFORE closing — re-using the open
        # connection and the in-memory cols saves a second JSON read
        # plus a second sqlite3.connect on every POST /api/marks.
        state_after = _state_map_from(cols, job_id, conn=conn)
    finally:
        conn.close()
    return state_after


# ──────────────────────────────────────────────────────────────────────
# App factory.
# ──────────────────────────────────────────────────────────────────────

def create_app(drawing_id: str, config: dict) -> FastAPI:
    """Build the FastAPI app for `drawing_id`. Raises FileNotFoundError
    early if the drawing is not ingested OR its DZI tile pyramid is
    missing — the spec mandates a loud failure rather than a silent
    single-bitmap fallback."""
    raster_path, meta = resolve_drawing(drawing_id)
    if meta.get("dzi_path") is None or not Path(meta["dzi_path"]).exists():
        raise FileNotFoundError(
            f"DZI tile pyramid for drawing_id={drawing_id!r} is missing. "
            f"Run:\n  python3 scripts/hitl.py build-tiles {drawing_id}"
        )

    _ensure_sidecar_tables()

    app = FastAPI(title=f"column-review · {drawing_id}", docs_url=None,
                  redoc_url=None)

    # ── static files (the OSD frontend + vendored OpenSeadragon) ──
    app.mount("/static",
              StaticFiles(directory=str(STATIC_DIR), html=False),
              name="static")

    # ── DZI manifest + tiles (scoped strictly to this drawing) ──
    @app.get("/dzi/{path:path}")
    def serve_dzi(path: str):
        # Path safety: only allow paths under `<drawing_id>.dzi` and
        # `<drawing_id>_files/...`. Reject any other access (e.g.
        # `..` traversal or sibling drawings).
        if not (path == f"{drawing_id}.dzi"
                or path.startswith(f"{drawing_id}_files/")):
            raise HTTPException(status_code=404)
        target = RAW_DRAWINGS_DIR / path
        # Guard against path traversal beyond RAW_DRAWINGS_DIR.
        try:
            target.resolve().relative_to(RAW_DRAWINGS_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=404)
        if not target.exists():
            raise HTTPException(status_code=404)
        return FileResponse(str(target))

    @app.head("/api/dzi-exists")
    def dzi_exists():
        ok = Path(meta["dzi_path"]).exists()
        return Response(status_code=200 if ok else 404)

    # ── root index ──
    @app.get("/", response_class=HTMLResponse)
    def index():
        idx = STATIC_DIR / "index.html"
        if not idx.exists():
            return HTMLResponse(
                "<h1>correction-app frontend missing</h1>"
                f"<p>Expected at {idx}</p>",
                status_code=500,
            )
        return HTMLResponse(idx.read_text())

    # ── session bootstrap ──
    # session_id and reviewer_id are kept in mutable holders so the
    # /api/session POST can replace them without re-creating the app.
    # /api/drawing and /api/config always read fresh from these holders
    # (and from `~/.column-review.json` via _load_reviewer_id), so a
    # browser refresh after the prompt is submitted sees the new value
    # rather than the stale closure capture noted in the code-review.
    session_id: dict = {"id": None}
    initial_reviewer_id = _load_reviewer_id()
    if initial_reviewer_id:
        session_id["id"] = _start_session(initial_reviewer_id)

    job_id = _find_or_create_job(drawing_id, raster_path,
                                 source_path=str(raster_path.resolve()))

    def _current_reviewer_id() -> Optional[str]:
        # ALWAYS re-read from disk. Closure capture would go stale after
        # POST /api/session updates ~/.column-review.json.
        return _load_reviewer_id()

    def _require_session() -> str:
        # Marking is blocked until the reviewer-id prompt is submitted.
        # Spec mandates per-mark provenance via reviewer_sessions; an
        # empty session_id would leave orphan tp_confirmations rows
        # that reference no row in reviewer_sessions.
        sid = session_id["id"]
        if not sid:
            raise HTTPException(
                status_code=409,
                detail="reviewer_id is not set. Submit the reviewer-id "
                       "prompt (POST /api/session) before marking.",
            )
        return sid

    # ── api: drawing bootstrap ──
    @app.get("/api/drawing")
    def get_drawing():
        det_path = JOBS_DIR / job_id / "px_detections.json"
        detections = (json.loads(det_path.read_text())
                      if det_path.exists() else {"columns": [], "meta": {}})
        return JSONResponse({
            "drawing_id":  drawing_id,
            "dzi_url":     f"/dzi/{drawing_id}.dzi",
            "raster_size": meta.get("size"),
            "job_id":      job_id,
            "session_id":  session_id["id"],
            "reviewer_id": _current_reviewer_id(),
            "detections":  detections,
            "config":      config,
        })

    @app.get("/api/state")
    def get_state():
        return JSONResponse(_build_state_map(job_id))

    # ── api: marks ──
    # Both single and batch routes funnel through `_apply_marks` so a
    # rubber-band of 500 marks commits as ONE SQLite transaction (one
    # fsync), not N.
    @app.post("/api/marks")
    async def post_mark(req: Request):
        sid = _require_session()
        body = await req.json()
        # _apply_marks returns the post-apply state map from the same
        # in-memory cols + open SQLite conn — avoids a second JSON read
        # and a second DB connect per mark (code-review #6).
        state = _apply_marks([body], job_id, sid)
        return JSONResponse({"ok": True, "state": state})

    @app.post("/api/marks/batch")
    async def post_marks_batch(req: Request):
        sid = _require_session()
        body = await req.json()
        marks = body.get("marks", [])
        if not isinstance(marks, list):
            raise HTTPException(status_code=400,
                                detail="`marks` must be a list")
        state = _apply_marks(marks, job_id, sid)
        return JSONResponse({"ok": True, "n": len(marks), "state": state})

    # ── api: session ──
    @app.post("/api/session")
    async def post_session(req: Request):
        body = await req.json()
        new_id = (body.get("reviewer_id") or "").strip()
        if not new_id:
            raise HTTPException(status_code=400,
                                detail="reviewer_id is required")
        _save_reviewer_id(new_id)
        session_id["id"] = _start_session(new_id)
        return JSONResponse({"reviewer_id": new_id,
                             "session_id": session_id["id"]})

    # ── api: config (frontend reads on boot) ──
    @app.get("/api/config")
    def get_config():
        return JSONResponse({
            "tile_cache_mb":    config.get("tile_cache_mb", 512),
            "hit_tolerance_px": config.get("hit_tolerance_px", 8),
            "snap_grid_px":     config.get("snap_grid_px", 0),
            "reviewer_id":      _load_reviewer_id(),
        })

    return app


# ──────────────────────────────────────────────────────────────────────
# Launcher helpers (called from scripts/hitl.py review).
# ──────────────────────────────────────────────────────────────────────

def pick_port(start: int, attempts: int = 20) -> int:
    """Return the first free loopback TCP port in [start, start+attempts)."""
    for p in range(start, start + attempts):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
            continue
    raise RuntimeError(
        f"No free port in [{start}, {start + attempts}). "
        "Pass --port to retry from a different base."
    )


def open_browser_soon(url: str, delay_seconds: float = 1.5) -> None:
    """Open the browser after `delay_seconds`, on a daemon thread so it
    doesn't block uvicorn's foreground run."""
    def _open():
        time.sleep(delay_seconds)
        try:
            webbrowser.open(url)
        except Exception:
            pass   # browser open is a nicety, not a correctness path
    threading.Thread(target=_open, daemon=True).start()
