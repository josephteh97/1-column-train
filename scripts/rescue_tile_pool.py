"""Unified rescue-YOLO tile pool — turns FP + FN_ADDED + TP-confirm
corrections into a single on-disk training pool that survives ⌫ Clear
detections.

Replaces the two crop-shaped pools (`hard_negatives/` + `fn_positives/`)
with one tile-shaped pool the rescue YOLO consumes directly. YOLO's
standard "missing label = no column here" supervision means FP
corrections need no special handling — they're just empty `.txt`
files alongside positive tiles in the same directory.

Layout written:
    data/rescue_tiles/images/<drawing-id>__<hash>.jpg
    data/rescue_tiles/labels/<drawing-id>__<hash>.txt
    data/rescue_tiles/manifest.json

manifest.json["entries"] is a list of dicts:
    {filename, drawing_id, job_id, kind, source_correction_ids[],
     tile_x0, tile_y0, tile_x1, tile_y1, timestamp}

`kind` is determined by the tile's CONTENTS, not by the originating
correction:
  - `fn_positive` — at least one accepted positive (TP-confirm or
    un-FP'd model-source detection or human_added FN) has its centre
    inside the tile. Label file contains one YOLO line per accepted
    positive.
  - `fp_negative` — no accepted positives in the tile. Label file is
    empty; YOLO learns "no column here" via label absence.

A given tile coordinate is therefore unambiguously one kind given the
underlying drawing state, regardless of which correction caused us to
look at it. Multiple corrections in the same tile coalesce into a
single entry whose `source_correction_ids` lists all of them.

Cap: `MAX_POOL_SIZE` (default 2000) entries, newest first by
correction timestamp.

Usage:
    python3 scripts/rescue_tile_pool.py            # refresh from DB
    python3 scripts/rescue_tile_pool.py --max 500
    python3 scripts/rescue_tile_pool.py --dry-run  # show what would change
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

from PIL import Image

# Real plans at 300 DPI exceed Pillow's default decompression-bomb
# ceiling (~89 Mpx). Trusted local files.
Image.MAX_IMAGE_PIXELS = None

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent

sys.path.insert(0, str(_SCRIPTS_DIR))

from tile_geometry import (   # noqa: E402
    TILE_SIZE, bbox_at_index, tile_origin_for_bbox,
)

DATA_ROOT      = _PROJECT_ROOT / "data"
JOBS_DIR       = DATA_ROOT / "jobs"
CORR_DB        = DATA_ROOT / "corrections.db"
POOL_DIR       = DATA_ROOT / "rescue_tiles"
IMAGES_DIR     = POOL_DIR / "images"
LABELS_DIR     = POOL_DIR / "labels"
MANIFEST_PATH  = POOL_DIR / "manifest.json"

MAX_POOL_SIZE_DEFAULT = 2000


def _crop_id(job_id: str, x0: int, y0: int) -> str:
    """Deterministic id for a (job_id, tile_origin) pair."""
    return hashlib.sha1(
        f"{job_id}::{x0}::{y0}".encode()
    ).hexdigest()[:12]


def _drawing_id_from_meta(meta: dict, fallback: str) -> str:
    src = (meta or {}).get("source")
    if src:
        return Path(src).stem
    return fallback


def _accepted_positives_in_tile(cols: list, fp_set: set[int],
                                tile: tuple[int, int, int, int]
                                ) -> list[list[float]]:
    """Every accepted positive whose centre falls inside `tile`.

    Accepted positive = the row's centre is inside the tile bounds AND
    its index is not in `fp_set`. Includes both `human_added` FN_ADDED
    rows and un-FP'd model-source rows — they're all what YOLO should
    learn to detect.
    """
    x0, y0, x1, y1 = tile
    out: list[list[float]] = []
    for i, row in enumerate(cols):
        if not isinstance(row, dict):
            continue
        if i in fp_set:
            continue
        bbox = row.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if x0 <= cx < x1 and y0 <= cy < y1:
            out.append([float(b) for b in bbox])
    return out


def _bbox_to_yolo_label(bbox: list[float], tile_origin: tuple[int, int]
                        ) -> str:
    """Convert an absolute-pixel bbox to a YOLO label line relative to
    the 1280×1280 tile (class id 0). Clamped to the tile edges before
    normalisation; degenerate clamps return ``"".
    """
    x0, y0 = tile_origin
    bx1 = max(x0,            bbox[0]) - x0
    by1 = max(y0,            bbox[1]) - y0
    bx2 = min(x0 + TILE_SIZE, bbox[2]) - x0
    by2 = min(y0 + TILE_SIZE, bbox[3]) - y0
    if bx2 <= bx1 or by2 <= by1:
        return ""
    cx = (bx1 + bx2) / 2 / TILE_SIZE
    cy = (by1 + by2) / 2 / TILE_SIZE
    bw = (bx2 - bx1)     / TILE_SIZE
    bh = (by2 - by1)     / TILE_SIZE
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _load_corrections_and_fp_sets(
) -> tuple[list[dict], dict[str, set[int]]]:
    """Single pass over `corrections.db`. Returns:
      - `corrections`: newest-first stream of correction descriptors
        (each row is one effective FN_ADDED, edit, FP, or tp_confirm)
      - `fp_by_job`: `{job_id: set(element_index)}` of effective FPs,
        consumed by the per-tile positive-set computation so we don't
        re-iterate corrections per job.

    Combining the two outputs avoids the previous "N+1 DB connections
    per pool rebuild" pattern — one connect, two products.
    """
    if not CORR_DB.exists():
        return [], {}
    from corrections_logger import iter_effective_corrections   # noqa: E402

    out: list[dict] = []
    fp_by_job: dict[str, set[int]] = {}
    conn = sqlite3.connect(str(CORR_DB))
    try:
        rowid_lookup = {
            (r[0], r[1], int(r[2])): r[3] for r in conn.execute(
                "SELECT job_id, element_type, element_index, id "
                "FROM corrections ORDER BY id"
            ).fetchall()
        }
        for r in iter_effective_corrections(conn):
            job_id, el_type, idx, _orig, _changes, is_delete, ts = r
            rid = rowid_lookup.get((job_id, el_type, int(idx)))
            if rid is None:
                continue
            if is_delete:
                fp_by_job.setdefault(job_id, set()).add(int(idx))
            out.append({
                "job_id":         job_id,
                "element_index":  int(idx),
                "timestamp":      float(ts),
                "correction_id":  f"c{rid}",
            })
        for rid, job_id, idx, ts in conn.execute(
            "SELECT rowid, job_id, element_index, ts FROM tp_confirmations"
        ).fetchall():
            out.append({
                "job_id":         job_id,
                "element_index":  int(idx),
                "timestamp":      float(ts),
                "correction_id":  f"t{rid}",
            })
    finally:
        conn.close()
    out.sort(key=lambda r: r["timestamp"], reverse=True)
    return out, fp_by_job


def _load_job_state(job_id: str, fp_set: set[int]
                    ) -> tuple[list, str, int, int] | None:
    """Return `(cols, drawing_id, W, H)` for a job, or None if the
    px_detections.json is missing or unparseable. `fp_set` is supplied
    by the caller (computed once in `_load_corrections_and_fp_sets`).
    """
    cols_path = JOBS_DIR / job_id / "px_detections.json"
    if not cols_path.is_file():
        return None
    try:
        d = json.loads(cols_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    cols = d.get("columns", [])
    meta = d.get("meta", {})
    drawing_id = _drawing_id_from_meta(meta, job_id)

    W = H = None
    size = meta.get("size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            W, H = int(size[0]), int(size[1])
        except (TypeError, ValueError):
            W = H = None
    if W is None or H is None:
        render_path = JOBS_DIR / job_id / "render.jpg"
        if not render_path.exists():
            return None
        with Image.open(render_path) as im:
            W, H = im.size
    return cols, drawing_id, W, H


def build_pool(max_size: int, dry_run: bool) -> dict:
    corrections, fp_by_job = _load_corrections_and_fp_sets()
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    capped = corrections[:max_size]

    # Group corrections by job so each render decodes exactly once.
    by_job: dict[str, list[dict]] = {}
    for c in capped:
        by_job.setdefault(c["job_id"], []).append(c)

    entries: list[dict] = []
    entries_by_id: dict[str, dict] = {}
    new_writes = 0
    skipped_missing_render = 0

    for job_id, job_corrs in by_job.items():
        render_path = JOBS_DIR / job_id / "render.jpg"
        fp_set    = fp_by_job.get(job_id, set())
        job_state = _load_job_state(job_id, fp_set)
        if job_state is None:
            skipped_missing_render += len(job_corrs)
            continue
        cols, drawing_id, W, H = job_state

        # Coalesce corrections by (tile_x0, tile_y0). For each unique
        # tile, build one entry whose source_correction_ids lists every
        # correction that contributed.
        tile_to_corrs: dict[tuple[int, int], list[dict]] = {}
        for c in job_corrs:
            bbox = bbox_at_index(cols, c["element_index"])
            if bbox is None:
                continue
            x0, y0 = tile_origin_for_bbox(bbox, W, H)
            tile_to_corrs.setdefault((x0, y0), []).append(c)

        # Decide which tiles actually need writing (skip if file
        # already on disk with matching label semantics).
        tiles_to_open: list[tuple[dict, int, int]] = []
        for (x0, y0), corrs in tile_to_corrs.items():
            cid       = _crop_id(job_id, x0, y0)
            tile_box  = (x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)
            positives = _accepted_positives_in_tile(cols, fp_set, tile_box)
            kind      = "fn_positive" if positives else "fp_negative"
            src_ids   = sorted({c["correction_id"] for c in corrs})
            newest_ts = max(c["timestamp"] for c in corrs)

            entry = {
                "filename":              f"{drawing_id}__{cid}.jpg",
                "drawing_id":            drawing_id,
                "job_id":                job_id,
                "kind":                  kind,
                "source_correction_ids": src_ids,
                "tile_x0":               x0,
                "tile_y0":               y0,
                "tile_x1":               x0 + TILE_SIZE,
                "tile_y1":               y0 + TILE_SIZE,
                "timestamp":             newest_ts,
            }
            entries_by_id[cid] = entry
            entries.append(entry)

            img_out = IMAGES_DIR / entry["filename"]
            lbl_out = LABELS_DIR / (Path(entry["filename"]).stem + ".txt")
            # Rewrite label file every time so newly-added/removed
            # positives at the same tile coords reflect on disk; the
            # label is cheap. The .jpg only writes when missing.
            if dry_run:
                continue
            if kind == "fp_negative":
                lbl_out.write_text("")
            else:
                lines = [_bbox_to_yolo_label(p, (x0, y0))
                         for p in positives]
                lines = [ln for ln in lines if ln]
                lbl_out.write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
            if not img_out.is_file():
                tiles_to_open.append((entry, x0, y0))

        if dry_run or not tiles_to_open:
            continue
        if not render_path.exists():
            skipped_missing_render += len(tiles_to_open)
            continue

        print(f"  cropping {len(tiles_to_open)} rescue tile(s) from job "
              f"{job_id[:8]}… ({drawing_id})", flush=True)
        with Image.open(render_path) as im:
            im.load()
            for entry, x0, y0 in tiles_to_open:
                tile = im.crop((x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE))
                img_out = IMAGES_DIR / entry["filename"]
                tile.convert("RGB").save(img_out, "JPEG", quality=85)
                new_writes += 1

    # Prune disk files no longer referenced (rescind safety).
    keep_image_files = {e["filename"] for e in entries}
    keep_label_files = {Path(f).stem + ".txt" for f in keep_image_files}
    pruned = 0
    if not dry_run:
        for p in IMAGES_DIR.glob("*.jpg"):
            if p.name not in keep_image_files:
                p.unlink()
                pruned += 1
        for p in LABELS_DIR.glob("*.txt"):
            if p.name not in keep_label_files:
                p.unlink()
                pruned += 1

    manifest = {
        "version":     1,
        "max_size":    max_size,
        "updated_ts":  time.time(),
        "n_entries":   len(entries),
        "entries":     entries,
    }
    if not dry_run:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    n_pos = sum(1 for e in entries if e["kind"] == "fn_positive")
    n_neg = sum(1 for e in entries if e["kind"] == "fp_negative")
    return {
        "n_entries":              len(entries),
        "n_positive":             n_pos,
        "n_negative":             n_neg,
        "new_writes":             new_writes,
        "pruned":                 pruned,
        "skipped_missing_render": skipped_missing_render,
        "dry_run":                dry_run,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max", type=int, default=MAX_POOL_SIZE_DEFAULT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    stats = build_pool(args.max, args.dry_run)
    print(f"Rescue-tile pool: {stats['n_entries']} entries "
          f"({stats['n_positive']} positive / "
          f"{stats['n_negative']} negative)")
    print(f"  new writes: {stats['new_writes']}, "
          f"pruned: {stats['pruned']}, "
          f"skipped missing render: {stats['skipped_missing_render']}")
    if stats["dry_run"]:
        print("(dry-run — no files written)")


if __name__ == "__main__":
    main()
