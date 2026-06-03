"""Human-in-the-loop corrections logger for the column detector.

Produces the inputs that `scripts/retrain_yolo.py` consumes:

    data/corrections.db                     — SQLite log of edits/deletes/adds
    data/jobs/{job_id}/render.jpg           — the plan image as reviewed
    data/jobs/{job_id}/px_detections.json   — { "columns": [{"bbox": [...], ...}, ...] }

The reviewer (a notebook UI) creates a job, dumps the post-processed
detections, then records one correction per disagreement:

    is_delete=True   → false positive (drop from labels at retrain time)
    is_delete=False  → bbox edit OR human-added missed detection

`element_index` indexes into the saved px_detections.json["columns"] list.
For human-added detections the entry is APPENDED to that list before the
correction row is written, so the index always points to a real entry.

Single-class detector: every correction uses element_type='column'.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Iterable

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("data")
JOBS_DIR   = DATA_ROOT / "jobs"
DB_PATH    = DATA_ROOT / "corrections.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT    NOT NULL,
    element_type     TEXT    NOT NULL,
    element_index    INTEGER NOT NULL,
    original_element TEXT    NOT NULL,   -- JSON
    changes          TEXT    NOT NULL,   -- JSON
    is_delete        INTEGER NOT NULL DEFAULT 0,
    timestamp        REAL    NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_corrections_job ON corrections(job_id);
-- UNIQUE on (job_id, element_index, is_delete) makes record_delete /
-- record_edit / record_add idempotent: clicking Save twice cannot
-- create duplicate rows for the same (job, detection, action) triple.
-- Adds use a different element_index per call, so they remain distinct.
CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_unique
    ON corrections(job_id, element_index, is_delete);
"""


class JobAlreadyCorrected(RuntimeError):
    """Raised when an operation would overwrite job state after
    corrections have already been logged for that job_id."""


def _ensure_dirs():
    DATA_ROOT.mkdir(exist_ok=True)
    JOBS_DIR.mkdir(exist_ok=True)


def _ensure_db() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def new_job_id() -> str:
    """Generate a fresh job_id (uuid4 hex)."""
    return uuid.uuid4().hex


def save_job(job_id: str, image, boxes, scores=None,
             source_path: str | None = None) -> Path:
    """Persist the reviewed plan + detections under data/jobs/{job_id}/.

    Raises JobAlreadyCorrected if the job already has at least one
    correction row — overwriting px_detections.json after corrections
    exist would silently invalidate element_index references.

    Parameters
    ----------
    job_id      : returned by new_job_id() (or any unique string).
    image       : PIL.Image — the FULL plan image (will be saved as render.jpg).
    boxes       : iterable of (x1, y1, x2, y2) in pixel coords.
    scores      : iterable of float confidences (optional, same length as boxes).
    source_path : original file path; recorded in px_detections.json["meta"].
    """
    _ensure_dirs()
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM corrections WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0]
        except sqlite3.OperationalError:
            n = 0   # table not yet created
        conn.close()
        if n > 0:
            raise JobAlreadyCorrected(
                f"job_id={job_id} already has {n} correction(s); refusing to "
                "overwrite data/jobs/{job_id}/px_detections.json. "
                "Use a fresh job_id (call new_job_id() again) for a new review."
            )
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    render_path = job_dir / "render.jpg"
    image.convert("RGB").save(render_path, quality=92, optimize=True)

    boxes_list = [list(map(float, b)) for b in boxes]
    if scores is None:
        scores_list = [1.0] * len(boxes_list)
    else:
        scores_list = [float(s) for s in scores]

    detections = {
        "columns": [
            {"bbox": b, "score": s}
            for b, s in zip(boxes_list, scores_list)
        ],
        "meta": {
            "source": source_path,
            "created_ts": time.time(),
            "n": len(boxes_list),
        },
    }
    (job_dir / "px_detections.json").write_text(json.dumps(detections, indent=2))
    return job_dir


def _load_px_detections(job_id: str) -> dict:
    path = JOBS_DIR / job_id / "px_detections.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _write_px_detections(job_id: str, det: dict):
    (JOBS_DIR / job_id / "px_detections.json").write_text(json.dumps(det, indent=2))


def record_delete(job_id: str, element_index: int):
    """Mark detection at `element_index` as a false positive (drop at retrain).

    Idempotent: calling this twice with the same (job_id, element_index)
    is a no-op (UNIQUE constraint + INSERT OR IGNORE)."""
    det = _load_px_detections(job_id)
    cols = det["columns"]
    if not (0 <= element_index < len(cols)):
        raise IndexError(f"element_index {element_index} out of range (n={len(cols)})")
    original = cols[element_index]
    conn = _ensure_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO corrections "
            "(job_id, element_type, element_index, original_element, changes, is_delete) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (job_id, "column", element_index, json.dumps(original), json.dumps({})),
        )
        conn.commit()
    finally:
        conn.close()


def record_edit(job_id: str, element_index: int, new_bbox):
    """Update the bbox of detection at `element_index`.

    A second edit on the same detection PRESERVES the original_element
    from the first edit's row (the model's true output), so the audit
    trail does not lose provenance. Only `changes` and `timestamp` are
    refreshed; `original_element` is sticky.

    The "edit retracts a prior delete" invariant is enforced at the
    read path in `scripts/retrain_yolo.build_dataset`, NOT here. We
    deliberately leave the prior is_delete=1 row in place so the audit
    trail records every reviewer action.

    Connection is closed in a finally block so a mid-call exception
    (disk full, locked DB) cannot leak the handle. JSON is mutated
    BEFORE the DB INSERT — if the DB write fails after the JSON write,
    retrain still uses the corrected bbox from px_detections.json; only
    the audit row is lost, not the training signal.
    """
    new_bbox = [float(x) for x in new_bbox]

    det = _load_px_detections(job_id)
    cols = det["columns"]
    if not (0 <= element_index < len(cols)):
        raise IndexError(f"element_index {element_index} out of range (n={len(cols)})")

    conn = _ensure_db()
    try:
        # Sticky-original: if a prior edit exists for this slot, preserve
        # its original_element (the model's TRUE output, captured before
        # any edit mutated px_detections.json). Without this, the second
        # edit would re-read from px_detections.json — which now holds
        # the first edit's bbox — and the model output is lost forever.
        row = conn.execute(
            "SELECT original_element FROM corrections "
            "WHERE job_id = ? AND element_index = ? AND is_delete = 0",
            (job_id, element_index),
        ).fetchone()
        original_json = row[0] if row is not None else json.dumps(dict(cols[element_index]))

        # Mutate JSON first (cheap, single-file write). Then write the DB
        # row. If the DB write fails after this point, retrain still uses
        # the corrected bbox from px_detections.json.
        cols[element_index]["bbox"] = new_bbox
        _write_px_detections(job_id, det)

        conn.execute(
            "INSERT OR REPLACE INTO corrections "
            "(job_id, element_type, element_index, original_element, changes, is_delete) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (job_id, "column", element_index, original_json,
             json.dumps({"bbox": new_bbox})),
        )
        conn.commit()
    finally:
        conn.close()


def record_add(job_id: str, bbox, score: float = 1.0) -> bool:
    """Append a human-added column (missed by the model) to px_detections.json
    AND log it in corrections.db. The retrain script picks it up as a label.

    Idempotent on (job_id, round(cx), round(cy)): if a previous
    record_add for this job already placed a human-added detection
    within 1 px of the same centre, this call is a no-op. Returns True
    if a new entry was appended, False if the call was deduped.
    """
    bbox = [float(x) for x in bbox]
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    rcx, rcy = round(cx), round(cy)

    det = _load_px_detections(job_id)
    cols = det["columns"]

    # Dedup: if an existing human_added entry has the same rounded centre,
    # skip. Re-running the notebook cell with the same `missed` list does
    # not double-append.
    for existing in cols:
        if existing.get("source") != "human_added":
            continue
        ex_bbox = existing.get("bbox") or []
        if len(ex_bbox) < 4:
            continue
        ecx = (ex_bbox[0] + ex_bbox[2]) / 2.0
        ecy = (ex_bbox[1] + ex_bbox[3]) / 2.0
        if round(ecx) == rcx and round(ecy) == rcy:
            return False

    new_entry = {"bbox": bbox, "score": float(score), "source": "human_added"}
    cols.append(new_entry)
    new_idx = len(cols) - 1
    _write_px_detections(job_id, det)
    conn = _ensure_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO corrections "
            "(job_id, element_type, element_index, original_element, changes, is_delete) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (job_id, "column", new_idx, json.dumps({}),
             json.dumps({"bbox": bbox, "source": "human_added"})),
        )
        conn.commit()
    finally:
        conn.close()
    return True


def summary() -> dict:
    """Return aggregate stats across the corrections DB."""
    if not DB_PATH.exists():
        return {"jobs": 0, "corrections": 0, "deletes": 0, "edits_or_adds": 0}
    conn = sqlite3.connect(str(DB_PATH))
    n_total   = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
    n_delete  = conn.execute("SELECT COUNT(*) FROM corrections WHERE is_delete = 1").fetchone()[0]
    n_jobs    = conn.execute("SELECT COUNT(DISTINCT job_id) FROM corrections").fetchone()[0]
    conn.close()
    return {
        "jobs": n_jobs,
        "corrections": n_total,
        "deletes": n_delete,
        "edits_or_adds": n_total - n_delete,
    }


if __name__ == "__main__":
    # Smoke test: roundtrip an empty job, no corrections, verify the DB exists.
    from PIL import Image
    jid = new_job_id()
    img = Image.new("RGB", (32, 32), (255, 255, 255))
    save_job(jid, img, [[1, 1, 10, 10]], [0.9], source_path="<smoke>")
    record_delete(jid, 0)
    print("Corrections summary:", summary())
