"""Read-only summary of `data/corrections.db`.

Run any time to see what's in the DB by drawing:
    python3 scripts/inspect_corrections.py

Uses the SAME rescind-on-read invariant every other consumer applies
(`iter_effective_corrections`), so the counts shown here match what
training will actually see.
"""
from __future__ import annotations

import datetime
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from column_review.path_bootstrap import ensure_on_path   # noqa: E402
ensure_on_path(_PROJECT_ROOT)

from scripts.corrections_logger import (   # noqa: E402
    DB_PATH,
    iter_effective_corrections,
    summary,
)


def _ts(t: float, fmt: str = "%m-%d %H:%M") -> str:
    return datetime.datetime.fromtimestamp(t).strftime(fmt)


def main() -> int:
    if not DB_PATH.exists():
        print(f"(no DB at {DB_PATH})")
        return 0

    print(f"DB: {DB_PATH}")
    print(f"Size: {DB_PATH.stat().st_size / 1024:.1f} KB")
    print()

    # Per-drawing breakdown — group the rescind-filtered stream in Python
    # so callers can't see a delete count that training will then ignore.
    conn = sqlite3.connect(str(DB_PATH))
    per_job: dict[str, dict] = defaultdict(
        lambda: {"fn": 0, "fp": 0, "first": float("inf"), "last": 0.0}
    )
    for job_id, _etype, _idx, _orig, _chg, is_delete, ts in (
        iter_effective_corrections(conn)
    ):
        agg = per_job[job_id]
        if is_delete:
            agg["fp"] += 1
        else:
            agg["fn"] += 1
        if ts < agg["first"]:
            agg["first"] = ts
        if ts > agg["last"]:
            agg["last"] = ts

    print("=== Corrections by drawing (rescind-filtered) ===")
    if not per_job:
        print("  (no corrections)")
    for job_id, agg in sorted(per_job.items(), key=lambda kv: -kv[1]["last"]):
        print(f"  {job_id[:8]}…  FN/edits={agg['fn']:<4}  "
              f"FPs={agg['fp']:<4}  "
              f"span={_ts(agg['first'])} -> {_ts(agg['last'])}")
    print(f"\n  TOTAL effective rows: "
          f"{sum(a['fn'] + a['fp'] for a in per_job.values())}")

    print()
    print("=== summary() ===")
    print(f"  {summary()}")

    print()
    print("=== TP confirmations ===")
    try:
        tp_rows = conn.execute(
            "SELECT job_id, COUNT(*) FROM tp_confirmations GROUP BY job_id"
        ).fetchall()
        if not tp_rows:
            print("  (none)")
        for j, n in tp_rows:
            print(f"  {j[:8]}…  {n} confirmations")
    except sqlite3.OperationalError:
        print("  (table not yet created)")

    print()
    print("=== Recent 10 corrections (raw, any drawing) ===")
    recent = conn.execute(
        """
        SELECT timestamp, job_id, element_index, is_delete, element_type
        FROM corrections
        ORDER BY timestamp DESC
        LIMIT 10
        """
    ).fetchall()
    for ts, j, idx, d, etype in recent:
        kind = "FP-delete" if d else "FN/edit  "
        print(f"  {_ts(ts, '%m-%d %H:%M:%S')}  {j[:8]}…  "
              f"idx={idx:<4}  {kind}  {etype}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
