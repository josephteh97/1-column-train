"""Human-in-the-loop corrections storage — schema + read helpers.

This module owns the on-disk shape that the retrain pipeline consumes:

    data/corrections.db                     — SQLite log of edits / deletes / adds
    data/jobs/{job_id}/render.jpg           — the plan image as reviewed
    data/jobs/{job_id}/px_detections.json   — { "columns": [{"bbox": [...], ...}, ...] }

Public surface (the only API used in production):

    new_job_id()                       → uuid hex; assign a fresh review session.
    iter_effective_corrections(conn, …) → yield correction rows with rescinded
                                          deletes filtered OUT. Single source of
                                          truth for "what corrections are live."
    summary()                          → aggregate counts (jobs, effective
                                          corrections, deletes, edits/adds,
                                          rescinded_deletes). Used by
                                          `hitl.py status`.

History: this module used to expose `save_job`, `record_delete`,
`record_edit`, `record_add`, and a `JobAlreadyCorrected` exception as
the storage WRITE path consumed by `correct_detections.ipynb`. The
notebook was deleted in change `rebuild-correction-ui-web`; its
replacement (the FastAPI web reviewer in `column_review/`, launched
via the top-level `column-review` CLI) inlines its writes into one
SQLite transaction per batch via `_apply_mark_locked`. The legacy
write helpers are gone with the notebook — the on-disk schema and
the read helpers above are the durable contract.

is_delete=True   → false positive (drop from labels at retrain time).
is_delete=False  → bbox edit OR human-added missed detection.

`element_index` indexes into the saved px_detections.json["columns"]
list. Single-class detector: every correction uses element_type='column'.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
# Anchor everything to the project root via __file__ — otherwise the DB
# and per-job folders are written relative to whatever CWD the caller
# happened to have, which makes the notebook and subprocess callers
# (hitl.py, train_bbox_classifier.py) see different files.
_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
DATA_ROOT  = _PROJECT_ROOT / "data"
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
-- UNIQUE on (job_id, element_index, is_delete) keeps the schema
-- idempotency invariant for any writer (notebook, web reviewer, future
-- batch tools): the same (job, detection, action) triple cannot
-- produce duplicate rows. Adds use a distinct element_index per call
-- so they remain individually addressable.
CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_unique
    ON corrections(job_id, element_index, is_delete);
"""


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


def iter_effective_corrections(conn: sqlite3.Connection,
                                job_id: str | None = None):
    """Yield correction rows with rescinded deletes filtered OUT.

    An is_delete=1 row is "rescinded" when an is_delete=0 row exists for
    the same (job_id, element_type, element_index). The rescind invariant
    lives at the SCHEMA boundary here so every reader — `summary()`,
    `scripts/hard_negative_pool.py`, `scripts/train_bbox_classifier`,
    `column_review.routes.detections._compute_states`, any future consumer — sees
    the same effective state. Reading the DB raw will overcount deletes
    (and silently mis-train).

    Yields tuples: (job_id, element_type, element_index,
                    original_element_json, changes_json, is_delete, ts)
    """
    where = "WHERE job_id = ?" if job_id is not None else ""
    params = (job_id,) if job_id is not None else ()
    rows = conn.execute(
        f"SELECT job_id, element_type, element_index, original_element, "
        f"       changes, is_delete, timestamp "
        f"FROM corrections {where} ORDER BY id",
        params,
    ).fetchall()

    # Build the rescind set: (job_id, element_type, element_index) where an
    # edit row exists. Any is_delete=1 row with the same key is dropped.
    edit_keys: set[tuple[str, str, int]] = {
        (r[0], r[1], r[2]) for r in rows if not r[5]
    }
    for r in rows:
        if r[5] and (r[0], r[1], r[2]) in edit_keys:
            continue   # rescinded delete — silently filtered
        yield r


def summary() -> dict:
    """Return aggregate stats across the corrections DB, with the
    rescind-on-read invariant applied so rescinded deletes do NOT inflate
    the delete count."""
    if not DB_PATH.exists():
        return {"jobs": 0, "corrections": 0, "deletes": 0,
                "edits_or_adds": 0, "rescinded_deletes": 0}
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Effective counts: filter rescinded deletes via the shared helper.
        n_total_effective = 0
        n_delete          = 0
        jobs_seen: set[str] = set()
        for row in iter_effective_corrections(conn):
            n_total_effective += 1
            jobs_seen.add(row[0])
            if row[5]:
                n_delete += 1
        # Raw delete count (for the rescinded count).
        n_delete_raw = conn.execute(
            "SELECT COUNT(*) FROM corrections WHERE is_delete = 1"
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "jobs":               len(jobs_seen),
        "corrections":        n_total_effective,
        "deletes":            n_delete,
        "edits_or_adds":      n_total_effective - n_delete,
        "rescinded_deletes":  n_delete_raw - n_delete,
    }
