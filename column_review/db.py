"""SQLite layer for column-review.

Reuses `scripts/corrections_logger.py` for the canonical `corrections`
table + read helpers (`new_job_id`, `iter_effective_corrections`,
`summary`, `DB_PATH`). Owns the `CREATE TABLE IF NOT EXISTS` statements
for the two sidecar tables (`tp_confirmations`, `reviewer_sessions`)
and the `retrain_jobs` table that the Save & Submit subprocess tracker
writes into.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# `scripts.corrections_logger` is the single source of truth for the
# `corrections` table shape AND for the DB path. Make the import robust
# to import-order — if `column_review.db` is imported before
# `cli.main` has placed PROJECT_ROOT on sys.path (e.g., a standalone
# test, `python -c`, or uvicorn picking up the module first), bootstrap
# sys.path here so the `from scripts.* import …` below still resolves.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from column_review.path_bootstrap import ensure_on_path   # noqa: E402
ensure_on_path(_PROJECT_ROOT)

from scripts.corrections_logger import (  # noqa: E402,F401  (re-exported)
    DB_PATH,
    iter_effective_corrections,
    new_job_id,
    summary,
)
from scripts.corrections_logger import _ensure_db  # noqa: E402


# Sidecar-table DDL. `CREATE TABLE IF NOT EXISTS` makes this safe to
# run against an existing DB that already has the tables.
_SIDECAR_DDL = """
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

_RETRAIN_JOBS_DDL = """
CREATE TABLE IF NOT EXISTS retrain_jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pid          INTEGER,
    started_ts   REAL,
    status       TEXT,
    finished_ts  REAL,
    stderr_tail  TEXT
);
CREATE INDEX IF NOT EXISTS idx_retrain_jobs_status
    ON retrain_jobs(status);
"""


# Canonical `corrections` DDL — mirror of `scripts/corrections_logger.py`'s
# `_SCHEMA`. We duplicate it (rather than import the private constant)
# only so the `--db-path` override below can run the same CREATE on
# arbitrary paths without monkey-patching `corrections_logger.DB_PATH`.
# Keep in sync if the upstream schema changes.
_CORRECTIONS_DDL = """
CREATE TABLE IF NOT EXISTS corrections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT    NOT NULL,
    element_type     TEXT    NOT NULL,
    element_index    INTEGER NOT NULL,
    original_element TEXT    NOT NULL,
    changes          TEXT    NOT NULL,
    is_delete        INTEGER NOT NULL DEFAULT 0,
    timestamp        REAL    NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_corrections_job
    ON corrections(job_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corrections_unique
    ON corrections(job_id, element_index, is_delete);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection to the corrections DB.

    For the default path, delegates to `corrections_logger._ensure_db()`
    so the canonical `corrections` table is created on first use. For
    a custom `db_path` (e.g., `--db-path` flag for tests), runs the
    same DDL on the override target — otherwise the first read against
    a fresh test DB fails with `no such table: corrections` because
    `_ensure_db()` only ever touches the default path.
    """
    if db_path is None:
        return _ensure_db()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_CORRECTIONS_DDL)
    conn.commit()
    return conn


def ensure_sidecar_tables(conn: sqlite3.Connection) -> None:
    """Create `tp_confirmations` and `reviewer_sessions` if absent.

    Idempotent (`CREATE TABLE IF NOT EXISTS`). Safe to call every
    server startup; no-op when tables already exist.
    """
    conn.executescript(_SIDECAR_DDL)


def ensure_retrain_jobs_table(conn: sqlite3.Connection) -> None:
    """Create `retrain_jobs` if absent (training-job tracker).
    Read/written by `column_review/retrain_jobs.py`.
    """
    conn.executescript(_RETRAIN_JOBS_DDL)
