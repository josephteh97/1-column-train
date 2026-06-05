"""Emit deterministic per-drawing train/val/test splits.

Reads `data/raw/drawings/*.meta.json` (one per ingested drawing), then
emits `data/splits/{train,val,test}.txt` — one drawing ID per line.

Assignment rule (deterministic and reproducible across runs):

    h = sha1(drawing_id) % 100
    h <  70  →  train
    h <  85  →  val
    else     →  test

Adding a new drawing only ever moves its OWN ID into one split; it
never reshuffles existing drawings. The hash split is the spec's
contract: per-drawing partition, no bbox leak across splits.

Usage:
    python3 scripts/split_drawings.py
    python3 scripts/split_drawings.py --train 80 --val 10  # 80/10/10 instead of 70/15/15
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

RAW_DRAWINGS_DIR = Path("data/raw/drawings")
SPLITS_DIR       = Path("data/splits")


def hash_pct(key: str) -> int:
    """Stable 0-99 bucket for any string key.

    Public so other scripts (e.g. `train_yolo_rescue.py`'s
    rescue_tiles train/val partition) share one canonical
    SHA1-based deterministic bucketing scheme — drift between
    bucket implementations is a footgun.
    """
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


# Backwards-compat alias for the previous private name.
_hash_pct = hash_pct


def assign_split(drawing_id: str, train_pct: int, val_pct: int) -> str:
    h = hash_pct(drawing_id)
    if h < train_pct:
        return "train"
    if h < train_pct + val_pct:
        return "val"
    return "test"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=int, default=70, help="Train percentage (default: 70)")
    p.add_argument("--val",   type=int, default=15, help="Val percentage (default: 15)")
    p.add_argument("--drawings-dir", type=Path, default=RAW_DRAWINGS_DIR)
    p.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    args = p.parse_args()

    if args.train + args.val >= 100:
        raise SystemExit(f"train+val must be <100 (leaves room for test); got {args.train + args.val}")

    if not args.drawings_dir.exists():
        raise SystemExit(f"{args.drawings_dir} not found — ingest some drawings first")

    drawing_ids = sorted({p.stem for p in args.drawings_dir.glob("*.meta.json")})
    if not drawing_ids:
        raise SystemExit(f"No .meta.json files found in {args.drawings_dir}")

    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for did in drawing_ids:
        buckets[assign_split(did, args.train, args.val)].append(did)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in buckets.items():
        (args.out_dir / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))

    print(f"Splits written to {args.out_dir}/  (train={args.train}% / val={args.val}% / test={100-args.train-args.val}%)")
    for split, ids in buckets.items():
        print(f"  {split:5s}: {len(ids):4d} drawings")


if __name__ == "__main__":
    main()
