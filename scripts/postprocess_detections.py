"""Post-process YOLO column detections from tiled inference on a real plan.

Three filters, all shape-based (no model needed), addressing the FP patterns
catalogued from real-plan inference:

  (1) aspect_filter        — drop bboxes with max(w,h)/min(w,h) > max_aspect.
                              Real columns are square-to-~2:1. Anything taller
                              than ~3:1 is text or a line stroke.
  (2) center_distance_nms  — merge any two detections whose CENTRES are within
                              `max_dist_px` of each other. This is the seam-
                              ghost killer: two predictions of the SAME
                              physical column from adjacent tiles always have
                              nearly coincident centres, regardless of how
                              much the bbox sizes drift between tiles. Runs
                              BEFORE IoU NMS because the bbox-shape signal
                              IoU relies on can be unreliable for seam ghosts.
  (3) cross_tile_nms       — IoU-based NMS at iou_thr=0.20 (lower than the
                              prior 0.30) as a backup for partial overlaps
                              that survived step 2.

Usage from a consumer that runs tiled inference on an A0 plan:

    from scripts.postprocess_detections import filter_detections

    raw = run_tiled_inference(model, plan_png)   # list of dicts
    clean = filter_detections(raw)               # defaults are recommended

Detection format:
    {"x1": int, "y1": int, "x2": int, "y2": int, "conf": float, "cls": int}

No heavyweight deps — stdlib only.
"""
from __future__ import annotations

import math


def _iou(a, b) -> float:
    """IoU of two boxes given as {x1,y1,x2,y2}."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(d) -> tuple[float, float]:
    return (d["x1"] + d["x2"]) / 2.0, (d["y1"] + d["y2"]) / 2.0


def aspect_filter(detections, max_aspect=2.5):
    """Drop detections whose bbox aspect ratio (long side / short side)
    exceeds `max_aspect`. Real columns are at most ~2:1; line-shaped FPs
    over text strokes commonly exceed 3:1."""
    kept = []
    for d in detections:
        w = max(1, d["x2"] - d["x1"])
        h = max(1, d["y2"] - d["y1"])
        aspect = max(w, h) / min(w, h)
        if aspect <= max_aspect:
            kept.append(d)
    return kept


def center_distance_nms(detections, max_dist_px=20):
    """Greedy dedup by CENTRE DISTANCE. Sort by confidence descending; for
    each detection, suppress any later one whose centre is within
    `max_dist_px` of an already-kept centre. This is the seam-ghost killer
    — two predictions of the same physical column from adjacent tiles always
    share a centre to within a handful of pixels, regardless of how their
    bbox shapes drift across the seam.

    Default 20 px is well under the closest-column-to-column spacing of
    ~190 px on real grids, and well over typical seam-ghost displacement
    of 5-15 px."""
    by_conf = sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)
    kept = []
    kept_centres = []
    for d in by_conf:
        cx, cy = _center(d)
        collide = False
        for kx, ky in kept_centres:
            if math.hypot(cx - kx, cy - ky) <= max_dist_px:
                collide = True
                break
        if not collide:
            kept.append(d)
            kept_centres.append((cx, cy))
    return kept


def cross_tile_nms(detections, iou_thr=0.20):
    """Greedy IoU-NMS across the full canvas: sort by confidence descending,
    then suppress any later box whose IoU with an already-kept box is above
    `iou_thr`. Runs after `center_distance_nms` as a backup for partial
    overlaps where the centres didn't quite coincide but the bboxes still
    refer to the same column."""
    by_conf = sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)
    kept = []
    for d in by_conf:
        if any(_iou(d, k) > iou_thr for k in kept):
            continue
        kept.append(d)
    return kept


def filter_detections(detections, max_aspect=2.5,
                       center_dist_px=20, nms_iou=0.20):
    """Apply the full pipeline: aspect filter → centre-distance dedup → IoU
    NMS. Defaults are the recommended values for tiled YOLO inference on
    A0 plans with 200-px tile overlap."""
    after_aspect = aspect_filter(detections, max_aspect=max_aspect)
    after_center = center_distance_nms(after_aspect, max_dist_px=center_dist_px)
    return cross_tile_nms(after_center, iou_thr=nms_iou)


# ── Tiny self-test (only runs when executed directly) ─────────────────────────
if __name__ == "__main__":
    # Line-shaped FP (aspect 80/30 ≈ 2.67 > 2.5) should be dropped.
    # Two overlapping boxes at IoU ~0.35 should collapse to the higher-conf one.
    test = [
        {"x1": 100, "y1": 100, "x2": 130, "y2": 180,
         "conf": 0.40, "cls": 0},          # aspect 2.67 → drop
        {"x1": 200, "y1": 200, "x2": 240, "y2": 240,
         "conf": 0.90, "cls": 0},          # canonical
        {"x1": 215, "y1": 210, "x2": 250, "y2": 250,
         "conf": 0.70, "cls": 0},          # near-coincident centre → drop
        {"x1": 400, "y1": 400, "x2": 440, "y2": 445,
         "conf": 0.60, "cls": 0},          # isolated → keep
    ]
    out = filter_detections(test)
    assert len(out) == 2, f"expected 2, got {len(out)}: {out}"
    assert out[0]["conf"] == 0.90
    assert out[1]["conf"] == 0.60

    # Seam-ghost case: centres 8 px apart but DIFFERENT bbox sizes (30 vs 40 px).
    # The previous iou=0.30 pipeline collapsed only when IoU happened to be
    # high enough; the new centre-distance step catches this regardless.
    seam = [
        {"x1": 1080, "y1": 200, "x2": 1110, "y2": 230,
         "conf": 0.85, "cls": 0},          # canonical from tile A (30 px)
        {"x1": 1085, "y1": 205, "x2": 1125, "y2": 245,
         "conf": 0.55, "cls": 0},          # seam ghost from tile B (40 px,
                                            # centre 8 px off, IoU ~0.45)
    ]
    out = filter_detections(seam)
    assert len(out) == 1, f"seam-ghost: expected 1, got {len(out)}: {out}"
    assert out[0]["conf"] == 0.85

    # Adjacent-but-distinct columns: centres 200 px apart → must NOT merge.
    adj = [
        {"x1": 100, "y1": 100, "x2": 140, "y2": 140,
         "conf": 0.90, "cls": 0},
        {"x1": 300, "y1": 100, "x2": 340, "y2": 140,
         "conf": 0.90, "cls": 0},          # 200 px to the right
    ]
    out = filter_detections(adj)
    assert len(out) == 2, f"adjacent: expected 2, got {len(out)}: {out}"

    print("self-test OK")
