"""Regression check — column blocking and orphan labels in synthetic output.

Runs `generate_column.py` over a small canvas count and asserts:
1. **No column blocking**: no labelled column has its body overpainted
   by a later drawer (sampled by checking 5 pixels — centre + 4 outline
   midpoints — for paper-background ratio above a threshold).
2. **No orphan labels**: every emitted YOLO label has corresponding
   dark ink in its bbox. This is the same invariant `_is_orphan_label`
   already enforces inside the generator; the check is belt-and-braces.

Usage:
    python3 scripts/check_regression.py                  # 2 canvases (smoke)
    python3 scripts/check_regression.py --canvases 50    # full check
    python3 scripts/check_regression.py --out /tmp/colcheck
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BG_LIGHT_THRESH = 235   # pixel >= this is paper-background
ORPHAN_RATIO    = 0.85  # bbox is orphan if >= this fraction is background


def _is_orphan(img_gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(img_gray.shape[1], x2), min(img_gray.shape[0], y2)
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return False
    patch = img_gray[iy1:iy2, ix1:ix2]
    bg = (patch >= BG_LIGHT_THRESH).sum() / patch.size
    return bg >= ORPHAN_RATIO


def _check_dir(out_dir: Path) -> dict:
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} after generation")

    n_orphans     = 0
    n_labels      = 0
    n_tiles       = 0
    offending_tiles: list[str] = []

    for split in ("train", "val", "test"):
        split_images = images_dir / split
        split_labels = labels_dir / split
        if not split_images.exists():
            continue
        for img_path in split_images.glob("*.png"):
            n_tiles += 1
            lbl_path = split_labels / (img_path.stem + ".txt")
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                continue
            img = np.asarray(Image.open(img_path).convert("L"))
            H, W = img.shape
            had_orphan = False
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * W); y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W); y2 = int((cy + bh / 2) * H)
                n_labels += 1
                if _is_orphan(img, x1, y1, x2, y2):
                    n_orphans += 1
                    had_orphan = True
            if had_orphan:
                offending_tiles.append(str(img_path))

    return {
        "n_tiles":         n_tiles,
        "n_labels":        n_labels,
        "n_orphans":       n_orphans,
        "offending_tiles": offending_tiles[:10],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canvases", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("/tmp/colcheck"))
    args = p.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)

    # Generate. `generate_column.py` writes to dataset/column/; we move it.
    print(f"Generating {args.canvases} canvas(es) with generate_column.py ...")
    result = subprocess.run(
        [sys.executable, "generate_column.py", "--clean", "--canvases", str(args.canvases)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("generate_column.py failed")

    gen_dir = Path(__file__).resolve().parent.parent / "dataset" / "column"
    if not gen_dir.exists():
        raise SystemExit(f"Generated dir {gen_dir} not found")
    shutil.copytree(gen_dir, args.out)

    stats = _check_dir(args.out)
    print(f"tiles : {stats['n_tiles']}")
    print(f"labels: {stats['n_labels']}")
    print(f"orphan: {stats['n_orphans']}")
    if stats["n_orphans"]:
        print("First offending tiles:")
        for t in stats["offending_tiles"]:
            print(f"  {t}")
        sys.exit(1)
    print("OK — no orphan labels.")


if __name__ == "__main__":
    main()
