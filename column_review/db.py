"""SQLite layer for column-review.

Reuses `scripts/corrections_logger.py` for the canonical `corrections`
table + read helpers (`new_job_id`, `iter_effective_corrections`,
`summary`, `DB_PATH`). Owns the `CREATE TABLE IF NOT EXISTS` statements
for the two sidecar tables (`tp_confirmations`, `reviewer_sessions`)
and the new `retrain_jobs` table — these used to live in the deleted
`scripts/correction_app/app.py` and move here so this package is
self-sufficient.
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
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.corrections_logger import (  # noqa: E402,F401  (re-exported)
    DB_PATH,
    iter_effective_corrections,
    new_job_id,
    summary,
)
from scripts.corrections_logger import _ensure_db  # noqa: E402


# Sidecar-table DDL — moved verbatim from the deleted
# `scripts/correction_app/app.py:SIDECAR_DDL` so the column shapes are
# preserved exactly. `CREATE TABLE IF NOT EXISTS` makes this safe to
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


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection to the corrections DB.

    Delegates to `corrections_logger._ensure_db()` for the default path
    so the `corrections` table + indexes are created on first use.
    `db_path` override is for tests; production always uses the default.
    """
    if db_path is None:
        return _ensure_db()
    conn = sqlite3.connect(str(db_path))
    return conn


def ensure_sidecar_tables(conn: sqlite3.Connection) -> None:
    """Create `tp_confirmations` and `reviewer_sessions` if absent.

    Idempotent (`CREATE TABLE IF NOT EXISTS`). Safe to call every
    server startup; no-op when tables already exist.
    """
    conn.executescript(_SIDECAR_DDL)


def ensure_retrain_jobs_table(conn: sqlite3.Connection) -> None:
    """Create `retrain_jobs` if absent (Save & Submit job tracker).

    Schema is owned by this package — no equivalent in the deleted
    correction_app. Tranche D's `retrain_jobs.py` reads/writes here.
    """
    conn.executescript(_RETRAIN_JOBS_DDL)
