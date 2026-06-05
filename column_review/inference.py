"""YOLO inference for a single drawing — pure compute, no FastAPI.

`run_inference(drawing_id, raster_path, weights_path, config)` reads the
raster, runs `tiled_predict` + `run_pipeline`, returns the list of
detection dicts ready to be merged into `px_detections.json["columns"]`.
The caller (`routes/detections.py::post_infer`) handles JSON read /
write + concurrent-write locking — keeping that out of here lets the
inference path stay testable in isolation.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Lazy-loaded YOLO weights cache. Cache key is (path, mtime, size) so a
# `cp retrained_column_detection.pt column_detect.pt` promotion invalidates
# the cache without needing a server restart. Lock serialises concurrent
# first-load attempts (FastAPI's sync handlers run on a starlette
# threadpool, so two requests can race past the cache-miss check).
_MODEL_CACHE: dict = {"path": None, "mtime": None, "size": None,
                       "model": None}
_MODEL_LOCK = threading.Lock()


@dataclass
class InferenceResult:
    """Boxes / scores / per-tile counts from one inference pass.

    `boxes` is a list of [x1, y1, x2, y2] in image pixel coordinates.
    `scores` is the matching confidence list. `sources` carries each
    surviving box's origin in the two-YOLO union:
      `"detect"`  — only the frozen main `column_detect.pt` proposed
      `"rescue"`  — only the trainable `column_rescue.pt` proposed
      `"both"`    — both proposed in the same region; one survived
                    cross-detector NMS
    Used for telemetry and for the `source` field in the persisted
    `px_detections.json["columns"][i]`.

    `tile_counts` is the per-tile detection count from the main YOLO,
    surfaced for OOD spread checks (the rescue YOLO's per-tile counts
    are not currently tracked separately).

    `rescue_version` is the mtime epoch seconds of `column_rescue.pt`
    at inference time, or `None` when the rescue weights were absent
    (soft-fall back to main-detector-only output). The caller writes
    it into `meta.rescue_version` for the post-process pipeline cache
    key.
    """
    boxes: list[list[float]]
    scores: list[float]
    sources: list[str]
    tile_counts: list[int]
    device: str
    elapsed_seconds: float
    rescue_version: Optional[float] = None


def _get_or_load_model(weights_path: Path):
    """Return a cached `ultralytics.YOLO` for `weights_path`.

    First call pays the import + load cost (~2–5 s on CPU); subsequent
    calls are constant-time. Cache key includes mtime + size so a
    weights swap (e.g., promoting a fine-tuned checkpoint) reloads.
    """
    st = weights_path.stat()
    mtime, size = st.st_mtime, st.st_size
    with _MODEL_LOCK:
        cache = _MODEL_CACHE
        if (cache["model"] is not None
                and cache["path"] == str(weights_path)
                and cache["mtime"] == mtime
                and cache["size"] == size):
            return cache["model"]
        print(f"[infer] loading weights {weights_path.name}…", flush=True)
        from ultralytics import YOLO
        cache["model"] = YOLO(str(weights_path))
        cache["path"] = str(weights_path)
        cache["mtime"] = mtime
        cache["size"] = size
        return cache["model"]


def _auto_device(explicit: Optional[str]) -> str:
    """Pick `cuda:0` if available, else `cpu`. Print the choice once so
    the user knows whether they're paying for GPU or CPU inference."""
    if explicit:
        return explicit
    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"[infer] auto-selected device={device} "
          f"(--device flag was not set)", flush=True)
    return device


def run_inference(drawing_id: str, raster_path: Path,
                  weights_path: Path, config: dict) -> InferenceResult:
    """Run tiled YOLO inference + post-processing on one drawing.

    Reads the raster via PIL (MAX_IMAGE_PIXELS is unset by the caller —
    A0 at 300 DPI trips PIL's default decompression-bomb guard). Returns
    the post-processed bboxes in image pixel coordinates ready to be
    written into `px_detections.json["columns"]`.

    Pre-conditions checked by the caller:
    - `raster_path` exists (else the caller surfaces an `ingest` hint).
    - `weights_path` is a regular file (else the caller 500s with
       a useful diagnostic).
    """
    # PIL + ultralytics are big imports; defer them so module-import
    # cost is cheap and `column-review --help` stays snappy.
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import numpy as np
    # `scripts.` imports require the project root on sys.path — the CLI
    # arranges that at startup, so this is just defence-in-depth.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.tiled_inference import tiled_predict
    from scripts.postprocess_pipeline import run_pipeline, DEFAULT_CONFIG

    tile_size = int(config.get("tile_size", 1280))
    tile_step = int(config.get("tile_step", 1080))
    conf_th = float(config.get("conf_th", 0.25))
    iou_th = float(config.get("iou_th", 0.45))
    input_dpi = int(config.get("input_dpi", 300))
    device = _auto_device(config.get("device"))

    t0 = time.perf_counter()
    print(f"[infer] loading raster {raster_path.name}…", flush=True)
    img = Image.open(raster_path).convert("RGB")
    model = _get_or_load_model(weights_path)

    # Progress-line cadence — target ~10 ticks across the whole run
    # so the terminal isn't silent for a 30–90 s inference.
    def _n_windows(extent: int, win: int, stride: int) -> int:
        if extent <= win:
            return 1
        return (extent - win + stride - 1) // stride + 1

    n_cols = _n_windows(img.width, tile_size, tile_step)
    n_rows = _n_windows(img.height, tile_size, tile_step)
    total_tiles = max(1, n_cols * n_rows)
    progress_every = max(1, total_tiles // 10)

    print(f"[infer] tiled_predict on {img.width}×{img.height} "
          f"(tile={tile_size} step={tile_step} conf={conf_th} "
          f"iou={iou_th} device={device} ~{total_tiles} tiles)…",
          flush=True)
    main_boxes, main_scores, tile_counts = tiled_predict(
        model, img,
        tile=tile_size, step=tile_step,
        conf=conf_th, iou=iou_th, device=device,
        progress_every=progress_every,
    )
    print(f"[infer] main raw detections: {len(main_boxes)}", flush=True)

    # OCR filter is off in the column-review path — pytesseract runs
    # one tesseract subprocess per surviving bbox sequentially, which
    # for ~1k raw detections balloons inference latency from seconds
    # to minutes. The CLI smoke-test still uses the default config.

    # Rescue YOLO (the second, trainable proposer in the two-YOLO
    # combined detector). Loads from <project_root>/column_rescue.pt by
    # default. Soft-fail by design: missing weights → empty output and
    # a one-shot stderr diagnostic, no exception. The cascade falls
    # back to main-detector-only output cleanly.
    from column_review.yolo_rescue import load_rescue
    rescue_weights = Path(
        config.get("rescue_weights")
        or project_root / "column_rescue.pt"
    )
    rescue_conf_th = float(config.get("rescue_conf_threshold", 0.4))
    rescue_version: float | None = None
    rescue_boxes: list[list[float]] = []
    rescue_scores: list[float] = []
    rescue_model, _ = load_rescue(rescue_weights, device=device)
    if rescue_model is not None:
        try:
            rescue_version = rescue_weights.stat().st_mtime
        except OSError:
            rescue_version = None
        print(f"[infer] rescue tiled_predict "
              f"({rescue_weights.name}, conf={rescue_conf_th})…", flush=True)
        rescue_boxes, rescue_scores, _rcounts = tiled_predict(
            rescue_model, img,
            tile=tile_size, step=tile_step,
            conf=rescue_conf_th, iou=iou_th, device=device,
            progress_every=progress_every,
        )
        print(f"[infer] rescue raw detections: {len(rescue_boxes)}",
              flush=True)
    else:
        print("[infer] rescue model absent → main-detector-only output",
              flush=True)

    # Union of detectors (stage 0 in the pipeline spec). Concatenate
    # main + rescue, then a single cross-detector NMS pass at the union
    # threshold. Source tags follow each surviving box; the NMS step
    # promotes any survivor that suppressed a different-source partner
    # to "both".
    union_iou = float(config.get("union_iou_threshold", 0.15))
    boxes, scores, sources = _union_detectors(
        main_boxes, main_scores,
        rescue_boxes, rescue_scores,
        iou_threshold=union_iou,
    )
    print(f"[infer] union: {len(main_boxes)} main + {len(rescue_boxes)} "
          f"rescue → {len(boxes)} after cross-detector NMS", flush=True)

    print("[infer] post-processing "
          f"({len(boxes)} merged → aspect → size → shape → "
          "centre-NMS → IoU-NMS)…",
          flush=True)
    img_gray = np.asarray(img.convert("L"))
    boxes_final, scores_final, audit = run_pipeline(
        img_gray, boxes, scores,
        config=DEFAULT_CONFIG,
        input_dpi=input_dpi,
        tile_detection_counts=tile_counts,
    )
    print(f"[infer] filtered detections: {len(boxes_final)}", flush=True)
    print(f"[infer] audit: {audit!r}", flush=True)

    # Re-attach source tags. `run_pipeline` filters rows but never
    # mutates coords, so an exact (x1,y1,x2,y2) match against the
    # pre-pipeline union recovers each survivor's source unambiguously.
    sources_final = _reattach_sources(boxes_final, boxes, sources)

    elapsed = time.perf_counter() - t0
    return InferenceResult(
        boxes=[[float(x) for x in bb] for bb in boxes_final.tolist()],
        scores=[float(s) for s in scores_final.tolist()],
        sources=sources_final,
        tile_counts=list(tile_counts) if tile_counts is not None else [],
        device=device,
        elapsed_seconds=elapsed,
        rescue_version=rescue_version,
    )


def _union_detectors(main_boxes: list, main_scores: list,
                     rescue_boxes: list, rescue_scores: list,
                     *, iou_threshold: float
                     ) -> tuple[list, list, list[str]]:
    """Cross-detector union via `torchvision.ops.nms` + source promotion.

    Concatenates the two detectors' outputs, runs vectorised NMS, then
    a single batched pairwise IoU pass identifies which survivors
    suppressed at least one different-source box (tagged `"both"`).
    Surviving boxes that didn't suppress a cross-detector partner keep
    their original tag (`"detect"` or `"rescue"`).
    """
    n_main, n_rescue = len(main_boxes), len(rescue_boxes)
    if n_main + n_rescue == 0:
        return [], [], []

    import torch
    import torchvision.ops as tvops

    all_boxes  = list(main_boxes) + list(rescue_boxes)
    all_scores = list(main_scores) + list(rescue_scores)
    src_tags   = ["detect"] * n_main + ["rescue"] * n_rescue

    boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    keep_idx = tvops.nms(boxes_t, scores_t,
                          iou_threshold=iou_threshold).tolist()
    keep_set = set(keep_idx)

    # Promote survivors that overlap any different-source non-survivor.
    # box_iou is N×M vectorised — one matrix multiply, no Python loop
    # over pairs. iou > threshold is the same suppression criterion the
    # NMS pass already used.
    if keep_idx and len(keep_idx) != len(all_boxes):
        kept_t  = boxes_t[keep_idx]
        suppr_idx = [i for i in range(len(all_boxes)) if i not in keep_set]
        suppr_t = boxes_t[suppr_idx]
        ious = tvops.box_iou(kept_t, suppr_t)   # (n_kept, n_suppressed)
        cross_hits = ious > iou_threshold
        for row, ki in enumerate(keep_idx):
            if not cross_hits[row].any():
                continue
            ki_src = src_tags[ki]
            for col, si in enumerate(suppr_idx):
                if cross_hits[row, col] and src_tags[si] != ki_src:
                    src_tags[ki] = "both"
                    break

    out_boxes  = [all_boxes[i]  for i in keep_idx]
    out_scores = [all_scores[i] for i in keep_idx]
    out_src    = [src_tags[i]   for i in keep_idx]
    return out_boxes, out_scores, out_src


def _reattach_sources(boxes_final, union_boxes: list,
                      union_sources: list[str]) -> list[str]:
    """Recover each post-pipeline survivor's source tag by exact-tuple
    lookup against the pre-pipeline union.

    `run_pipeline` filters rows but never mutates bbox coords, so the
    output values are identical to the corresponding union entries
    (numpy array roundtrip preserves float exactness for the values we
    care about). Survivors that find no match (shouldn't happen in
    normal flow) default to `"detect"` — the source tag is telemetry,
    not load-bearing on correctness.
    """
    lookup = {tuple(b): s for b, s in zip(union_boxes, union_sources)}
    return [lookup.get(tuple(b), "detect") for b in boxes_final.tolist()]
