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
# `cp column_detect_ft_<ts>.pt column_detect.pt` promotion invalidates
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
    `scores` is the matching confidence list. `tile_counts` is the
    per-tile detection count, surfaced for diagnostics only.
    """
    boxes: list[list[float]]
    scores: list[float]
    tile_counts: list[int]
    device: str
    elapsed_seconds: float


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
    boxes, scores, tile_counts = tiled_predict(
        model, img,
        tile=tile_size, step=tile_step,
        conf=conf_th, iou=iou_th, device=device,
        progress_every=progress_every,
    )
    print(f"[infer] raw detections: {len(boxes)}", flush=True)

    # OCR filter is off in the column-review path — pytesseract runs
    # one tesseract subprocess per surviving bbox sequentially, which
    # for ~1k raw detections balloons inference latency from seconds
    # to minutes. The deployed weights' precision is high enough that
    # OCR's "text inside a bbox" rejection costs more than it saves.
    # The CLI smoke-test still uses the default config.
    from dataclasses import replace
    pp_config = replace(DEFAULT_CONFIG, use_ocr_filter=False)

    print("[infer] post-processing "
          f"({len(boxes)} raw → aspect → size → shape → centre-NMS → IoU-NMS)…",
          flush=True)
    img_gray = np.asarray(img.convert("L"))
    boxes_final, scores_final, audit = run_pipeline(
        img_gray, boxes, scores,
        config=pp_config,
        input_dpi=input_dpi,
        tile_detection_counts=tile_counts,
    )
    print(f"[infer] filtered detections: {len(boxes_final)}", flush=True)
    print(f"[infer] audit: {audit!r}", flush=True)

    elapsed = time.perf_counter() - t0
    return InferenceResult(
        boxes=[[float(x) for x in bb] for bb in boxes_final.tolist()],
        scores=[float(s) for s in scores_final.tolist()],
        tile_counts=list(tile_counts) if tile_counts is not None else [],
        device=device,
        elapsed_seconds=elapsed,
    )
