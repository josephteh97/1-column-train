"""Shared tile-geometry + bbox primitives.

Single home for the TILE_SIZE invariant, tile-origin centring math,
axis-aligned IoU, and the column-lookup helper used by every script
that touches `data/jobs/<id>/px_detections.json` + `data/rescue_tiles/`.

`TILE_SIZE` is the load-bearing constant of the whole project
(CLAUDE.md "Tile-size invariant"): it MUST equal `IMGSZ` in `train.py`
and `TILE_SIZE` in `generate_column.py`. Duplicating it across
scripts/ was the original drift surface — own it here.
"""
from __future__ import annotations

from typing import Sequence


TILE_SIZE = 1280


def tile_origin_for_bbox(bbox: Sequence[float], W: int, H: int
                         ) -> tuple[int, int]:
    """Centre a 1280×1280 tile on the bbox, clamped to canvas bounds.

    The clamp ensures the tile never extends off-canvas; corrections
    near the edge get an off-centre tile that still fits.
    """
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    x0 = max(0, min(W - TILE_SIZE, cx - TILE_SIZE // 2))
    y0 = max(0, min(H - TILE_SIZE, cy - TILE_SIZE // 2))
    return x0, y0


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Axis-aligned IoU on (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-6, ua + ub - inter)


def bbox_at_index(cols: list, idx: int) -> list[float] | None:
    """Return `cols[idx].bbox` as `[x1, y1, x2, y2]`, or None on any
    miss (out-of-range, non-dict, missing key, malformed shape).
    `cols` is the already-parsed `px_detections.json["columns"]` list.
    """
    if not (0 <= idx < len(cols)):
        return None
    row = cols[idx]
    if not isinstance(row, dict):
        return None
    bbox = row.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    return [float(x) for x in bbox]
