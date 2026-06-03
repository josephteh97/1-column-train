"""Hard-negative pool — turns FP corrections into training-time negatives.

Reads `data/corrections.db` for `is_delete=1` rows, locates the FP
bbox in the matching `data/jobs/{job_id}/render.jpg`, crops the region
with a configurable margin, and writes the crop to
`data/hard_negatives/<drawing-id>__<hash>.png`. Maintains a manifest
at `data/hard_negatives/manifest.json` so the retrain can include
every entry as a zero-label background image.

Pool is capped at `MAX_POOL_SIZE` (default 2000) entries — newest
first by correction timestamp. Older entries are pruned.

Usage:
    python3 scripts/hard_negative_pool.py            # refresh from current DB
    python3 scripts/hard_negative_pool.py --max 500
    python3 scripts/hard_negative_pool.py --dry-run  # show what would change
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from PIL import Image

DATA_ROOT      = Path("data")
JOBS_DIR       = DATA_ROOT / "jobs"
CORR_DB        = DATA_ROOT / "corrections.db"
POOL_DIR       = DATA_ROOT / "hard_negatives"
MANIFEST_PATH  = POOL_DIR / "manifest.json"

MAX_POOL_SIZE_DEFAULT = 2000
CROP_MARGIN_PX        = 24


def _crop_id(job_id: str, element_index: int) -> str:
    """Deterministic id for a (job_id, element_index) pair."""
    return hashlib.sha1(f"{job_id}::{element_index}".encode()).hexdigest()[:12]


def _load_corrections_with_drawing() -> list[dict]:
    if not CORR_DB.exists():
        return []
    conn = sqlite3.connect(str(CORR_DB))
    rows = conn.execute(
        "SELECT job_id, element_index, original_element, timestamp "
        "FROM corrections WHERE is_delete = 1 ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    out = []
    for job_id, idx, original_json, ts in rows:
        try:
            original = json.loads(original_json)
        except json.JSONDecodeError:
            continue
        bbox = original.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        out.append({
            "job_id":       job_id,
            "element_index": int(idx),
            "bbox":         [float(x) for x in bbox],
            "timestamp":    float(ts),
        })
    return out


def _drawing_id_for_job(job_id: str) -> str:
    """Resolve drawing id for a job by reading its px_detections.json meta."""
    meta_path = JOBS_DIR / job_id / "px_detections.json"
    if not meta_path.exists():
        return job_id   # fallback
    try:
        d = json.loads(meta_path.read_text())
        src = d.get("meta", {}).get("source")
        if src:
            return Path(src).stem
    except (json.JSONDecodeError, OSError):
        pass
    return job_id


def build_pool(max_size: int, dry_run: bool) -> dict:
    corrections = _load_corrections_with_drawing()
    POOL_DIR.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    entries: list[dict] = []
    new_writes = 0
    skipped_missing_render = 0

    for c in corrections[:max_size]:
        cid = _crop_id(c["job_id"], c["element_index"])
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        drawing_id = _drawing_id_for_job(c["job_id"])
        out_name = f"{drawing_id}__{cid}.png"
        out_path = POOL_DIR / out_name

        if not out_path.exists():
            render_path = JOBS_DIR / c["job_id"] / "render.jpg"
            if not render_path.exists():
                skipped_missing_render += 1
                continue
            if not dry_run:
                with Image.open(render_path) as im:
                    W, H = im.size
                    x1, y1, x2, y2 = c["bbox"]
                    cx1 = max(0, int(x1) - CROP_MARGIN_PX)
                    cy1 = max(0, int(y1) - CROP_MARGIN_PX)
                    cx2 = min(W, int(x2) + CROP_MARGIN_PX)
                    cy2 = min(H, int(y2) + CROP_MARGIN_PX)
                    crop = im.crop((cx1, cy1, cx2, cy2))
                    crop.save(out_path, optimize=True)
                new_writes += 1

        entries.append({
            "filename":     out_name,
            "drawing_id":   drawing_id,
            "job_id":       c["job_id"],
            "source_bbox":  c["bbox"],
            "timestamp":    c["timestamp"],
        })

    # Prune stale png files that are no longer in the manifest.
    keep_files = {e["filename"] for e in entries}
    pruned = 0
    if not dry_run:
        for p in POOL_DIR.glob("*.png"):
            if p.name not in keep_files:
                p.unlink()
                pruned += 1

    manifest = {
        "version":      1,
        "max_size":     max_size,
        "updated_ts":   time.time(),
        "n_entries":    len(entries),
        "entries":      entries,
    }
    if not dry_run:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    return {
        "n_entries":            len(entries),
        "new_writes":           new_writes,
        "pruned":               pruned,
        "skipped_missing_render": skipped_missing_render,
        "dry_run":              dry_run,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max", type=int, default=MAX_POOL_SIZE_DEFAULT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    stats = build_pool(args.max, args.dry_run)
    print(f"Hard-negative pool: {stats['n_entries']} entries"
          f"  (new: {stats['new_writes']}, pruned: {stats['pruned']},"
          f" skipped missing render: {stats['skipped_missing_render']})")
    if stats["dry_run"]:
        print("(dry-run — no files written)")


if __name__ == "__main__":
    main()
