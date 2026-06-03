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

# Real plans at 300 DPI exceed Pillow's default decompression-bomb
# ceiling (~89 Mpx). Trusted local files, not adversarial uploads.
Image.MAX_IMAGE_PIXELS = None

# Anchor paths to project root via __file__ so cwd doesn't matter.
_SCRIPTS_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT  = _SCRIPTS_DIR.parent
DATA_ROOT      = _PROJECT_ROOT / "data"
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
    """Load FP corrections, filtering out rescinded deletes via the
    shared `iter_effective_corrections` helper. A delete-then-edit pair
    means the reviewer changed their mind — the original bbox MUST NOT
    become a hard-negative training crop (the reviewer's final intent
    was 'corrected real column', not 'false positive').
    """
    if not CORR_DB.exists():
        return []
    # Import lazily so the script imports clean if the logger isn't
    # available (e.g., during a partial deployment).
    from corrections_logger import iter_effective_corrections

    conn = sqlite3.connect(str(CORR_DB))
    try:
        # row layout from iter_effective_corrections:
        #   (job_id, element_type, element_index,
        #    original_element_json, changes_json, is_delete, timestamp)
        rows = [r for r in iter_effective_corrections(conn) if r[5]]
    finally:
        conn.close()
    # Newest first by timestamp (helper yields by insert order).
    rows.sort(key=lambda r: r[6], reverse=True)
    out = []
    for job_id, _el_type, idx, original_json, _changes_json, _is_del, ts in rows:
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

    # Cap first, THEN group by job_id so we open each large render exactly
    # once even when the same job contributes many FPs. Opening a 128 Mpx
    # plan once and cropping 172 times is seconds; opening it 172 times
    # is multi-minute and triggers the same bomb warning every iteration.
    capped = corrections[:max_size]
    by_job: dict[str, list[dict]] = {}
    for c in capped:
        by_job.setdefault(c["job_id"], []).append(c)

    seen_ids: set[str] = set()
    entries: list[dict] = []
    new_writes = 0
    skipped_missing_render = 0

    for job_id, job_corrs in by_job.items():
        drawing_id = _drawing_id_for_job(job_id)
        render_path = JOBS_DIR / job_id / "render.jpg"

        # Decide which crops actually need writing before touching the
        # image. If every corrected crop for this job is already on disk
        # OR every job_corrs entry was deduped against seen_ids, we can
        # skip opening the large render entirely.
        work: list[tuple[str, Path, list[float]]] = []   # (cid, out_path, bbox)
        for c in job_corrs:
            cid = _crop_id(c["job_id"], c["element_index"])
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            out_name = f"{drawing_id}__{cid}.png"
            out_path = POOL_DIR / out_name
            entries.append({
                "filename":     out_name,
                "drawing_id":   drawing_id,
                "job_id":       c["job_id"],
                "source_bbox":  c["bbox"],
                "timestamp":    c["timestamp"],
            })
            if not out_path.exists():
                work.append((cid, out_path, c["bbox"]))

        if not work:
            continue
        if not render_path.exists():
            skipped_missing_render += len(work)
            continue
        if dry_run:
            continue

        # Open once, decode once, crop many times. PIL load is implicit
        # on the first crop and cached for subsequent crops.
        print(f"  cropping {len(work)} hard-negative(s) from job "
              f"{job_id[:8]}… ({drawing_id})", flush=True)
        with Image.open(render_path) as im:
            W, H = im.size
            im.load()   # force a single decode of the 128 Mpx render
            for _cid, out_path, bbox in work:
                x1, y1, x2, y2 = bbox
                cx1 = max(0, int(x1) - CROP_MARGIN_PX)
                cy1 = max(0, int(y1) - CROP_MARGIN_PX)
                cx2 = min(W, int(x2) + CROP_MARGIN_PX)
                cy2 = min(H, int(y2) + CROP_MARGIN_PX)
                # Crop is fast on a loaded image; optimize=True on tiny
                # crops adds milliseconds — drop for further speedup.
                im.crop((cx1, cy1, cx2, cy2)).save(out_path)
                new_writes += 1

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
