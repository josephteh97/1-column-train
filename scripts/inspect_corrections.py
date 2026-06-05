"""Read-only summary of `data/corrections.db`.

Run any time to see what's in the DB by drawing:
    python3 scripts/inspect_corrections.py
"""
from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "data" / "corrections.db"


def main() -> None:
    if not _DB.exists():
        print(f"(no DB at {_DB})")
        return

    conn = sqlite3.connect(str(_DB))

    print(f"DB: {_DB}")
    print(f"Size: {_DB.stat().st_size / 1024:.1f} KB")
    print()

    print("=== Corrections by drawing ===")
    rows = conn.execute(
        """
        SELECT job_id,
               SUM(CASE WHEN is_delete=0 THEN 1 ELSE 0 END) AS fn_adds_or_edits,
               SUM(CASE WHEN is_delete=1 THEN 1 ELSE 0 END) AS fp_deletes,
               MIN(timestamp) AS first_ts,
               MAX(timestamp) AS last_ts
        FROM corrections
        GROUP BY job_id
        ORDER BY last_ts DESC
        """
    ).fetchall()
    if not rows:
        print("  (no corrections)")
    for j, fn, fp, t0, t1 in rows:
        first = datetime.datetime.fromtimestamp(t0).strftime("%m-%d %H:%M")
        last = datetime.datetime.fromtimestamp(t1).strftime("%m-%d %H:%M")
        print(f"  {j[:8]}…  FN/edits={fn:<4}  FPs={fp:<4}  span={first} -> {last}")

    total = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
    print(f"\n  TOTAL rows in corrections: {total}")

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
    print("=== Recent 10 corrections (any drawing) ===")
    recent = conn.execute(
        """
        SELECT timestamp, job_id, element_index, is_delete, element_type
        FROM corrections
        ORDER BY timestamp DESC
        LIMIT 10
        """
    ).fetchall()
    for ts, j, idx, d, etype in recent:
        when = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")
        kind = "FP-delete" if d else "FN/edit  "
        print(f"  {when}  {j[:8]}…  idx={idx:<4}  {kind}  {etype}")

    conn.close()


if __name__ == "__main__":
    main()
