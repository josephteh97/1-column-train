"""One-shot migration: archive the pre-rescue-yolo crop pools.

The unified `data/rescue_tiles/` pool is rebuilt from `corrections.db`
by `scripts/rescue_tile_pool.py`. The source of truth for which
corrections exist (and which were rescinded) lives in the DB, NOT in
the old crop manifests, so the migration's only data job is to ARCHIVE
the old crop directories — not to convert them.

What this script does:

  1. Archives `data/hard_negatives/` to
     `archive/pre-rescue-yolo/hard_negatives/` (if present).
  2. Archives `data/fn_positives/` to
     `archive/pre-rescue-yolo/fn_positives/` (if present).
  3. Archives `column_classifier.pt` and `column_classifier.meta.json`
     to `archive/pre-rescue-yolo/` (if present).
  4. Invokes `scripts/rescue_tile_pool.py` to materialise the new
     pool from `corrections.db`.

What this script does NOT do:

  - Convert old crop PNGs into rescue tiles. The DB is the source
    of truth; the rescue pool builder re-cuts from `render.jpg`
    using the actual bbox + tile geometry.
  - Delete anything not first moved into `archive/pre-rescue-yolo/`.
    The one-release archive policy keeps recovery cheap.

Idempotent: re-running after the archive already moved the
directories away is a no-op (the source paths are gone).

Usage:
    python3 scripts/migrate_pools_to_rescue_tiles.py
    python3 scripts/migrate_pools_to_rescue_tiles.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
DATA_ROOT     = _PROJECT_ROOT / "data"
ARCHIVE_ROOT  = _PROJECT_ROOT / "archive" / "pre-rescue-yolo"


def _archive(src: Path, dst: Path, dry_run: bool) -> str:
    """Move `src` to `dst`, returning a one-line status string. No-op
    if `src` is absent."""
    if not src.exists():
        return f"  skip   {src.relative_to(_PROJECT_ROOT)}: not present"
    if dst.exists():
        # Disambiguate the second migration run by appending a stamp.
        ts = int(time.time())
        dst = dst.with_name(f"{dst.name}_{ts}")
    if dry_run:
        return (f"  would archive {src.relative_to(_PROJECT_ROOT)} "
                f"→ {dst.relative_to(_PROJECT_ROOT)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"  moved  {src.relative_to(_PROJECT_ROOT)} → " \
           f"{dst.relative_to(_PROJECT_ROOT)}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change without moving files "
                        "or invoking the rescue pool builder.")
    args = p.parse_args()

    targets = [
        (DATA_ROOT / "hard_negatives",  ARCHIVE_ROOT / "hard_negatives"),
        (DATA_ROOT / "fn_positives",    ARCHIVE_ROOT / "fn_positives"),
        (_PROJECT_ROOT / "column_classifier.pt",
         ARCHIVE_ROOT / "column_classifier.pt"),
        (_PROJECT_ROOT / "column_classifier.meta.json",
         ARCHIVE_ROOT / "column_classifier.meta.json"),
    ]
    print("Archiving pre-rescue-yolo artefacts…")
    for src, dst in targets:
        print(_archive(src, dst, args.dry_run))

    print()
    print("Building rescue_tiles/ pool from corrections.db…")
    if args.dry_run:
        print("  (skipped — dry run)")
        return 0
    rc = subprocess.call(
        [sys.executable, str(_SCRIPTS_DIR / "rescue_tile_pool.py")],
        cwd=str(_PROJECT_ROOT),
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
