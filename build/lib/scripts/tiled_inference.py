"""Tiled YOLO inference helper.

Both notebooks and `scripts/retrain_yolo.py`'s regression evaluator
need to run inference on a full A0 plan by tiling it at the training
geometry (TILE_SIZE=1280, TILE_STEP=1080) and translating per-tile
detections to global coordinates. This is the one place where that
loop lives so that QA, deployed inference, and the audit regression
metric cannot drift.

The function returns BOTH the global boxes/scores AND the per-tile
raw detection counts — the OOD detector needs the per-tile spread,
not a global mean.

Out-of-bounds areas at the right and bottom edge of the plan (and the
entire output for plans smaller than the tile size) are filled with
WHITE (255), matching the synthetic training distribution. PIL's
default crop pads with black, which is wildly out-of-distribution for
a white-paper-trained model and produces spurious edge detections.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from PIL import Image


TILE_SIZE_DEFAULT = 1280
TILE_STEP_DEFAULT = 1080
WHITE_PAD = (255, 255, 255)


def _tile_grid(W: int, H: int, tile: int, step: int) -> tuple[list[int], list[int]]:
    xs = list(range(0, max(1, W - tile), step))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    ys = list(range(0, max(1, H - tile), step))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    return xs, ys


def tiled_predict(
    model,
    img,
    *,
    tile: int = TILE_SIZE_DEFAULT,
    step: int = TILE_STEP_DEFAULT,
    conf: float = 0.25,
    iou: float = 0.45,
    device=None,
    progress_every: int = 0,
) -> tuple[list[list[float]], list[float], list[int]]:
    """Run tiled inference on a PIL.Image at training geometry.

    Returns
    -------
    boxes      : list of [x1, y1, x2, y2] in GLOBAL image coords (floats).
    scores     : list of confidences parallel to boxes.
    tile_counts: list of per-tile raw detection counts (length =
                 n_tiles_x * n_tiles_y, ordered row-major).
                 Use this for OOD spread checks rather than the mean.
    """
    W, H = img.size
    xs, ys = _tile_grid(W, H, tile, step)

    boxes: list[list[float]] = []
    scores: list[float] = []
    tile_counts: list[int] = []

    # Resolve device once, here, so every caller gets correct CPU fallback.
    # If `device` is None, inherit from the model. Then coerce CUDA → CPU
    # when CUDA isn't available (handles the case where a CUDA-trained .pt
    # is loaded on a CPU-only retrain box and model.device still reports
    # cuda:0).
    if device is None:
        device = getattr(model, "device", None)
    if device is not None:
        try:
            import torch
            if "cuda" in str(device) and not torch.cuda.is_available():
                print(f"  tiled_predict: requested device={device} but CUDA "
                      "not available — falling back to CPU.", flush=True)
                device = "cpu"
        except ImportError:
            pass

    W_img, H_img = img.size
    n_total = len(xs) * len(ys)
    n_done = 0
    for ty in ys:
        for tx in xs:
            # Clip the actual crop to the image bounds, then paste it onto a
            # white tile-sized canvas. Out-of-bounds pixels are 255 (paper)
            # instead of PIL's default black — black bars off the right/bottom
            # edges produce spurious YOLO detections on a white-trained model.
            cx2 = min(tx + tile, W_img)
            cy2 = min(ty + tile, H_img)
            real = img.crop((tx, ty, cx2, cy2))
            tile_img = Image.new("RGB", (tile, tile), WHITE_PAD)
            tile_img.paste(real, (0, 0))
            kwargs = {
                "source":  tile_img,
                "imgsz":   tile,
                "conf":    conf,
                "iou":     iou,
                "verbose": False,
            }
            if device is not None:
                kwargs["device"] = device
            result = model.predict(**kwargs)[0]
            n_this_tile = 0
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                cfs  = result.boxes.conf.cpu().numpy()
                xyxy[:, [0, 2]] += tx
                xyxy[:, [1, 3]] += ty
                boxes.extend(xyxy.tolist())
                scores.extend(cfs.tolist())
                n_this_tile = len(xyxy)
            tile_counts.append(n_this_tile)
            n_done += 1
            if progress_every and n_done % progress_every == 0:
                print(f"  tiled_predict: {n_done}/{n_total} tiles", flush=True)

    return boxes, scores, tile_counts


def n_tiles_for_image(W: int, H: int,
                      tile: int = TILE_SIZE_DEFAULT,
                      step: int = TILE_STEP_DEFAULT) -> int:
    xs, ys = _tile_grid(W, H, tile, step)
    return len(xs) * len(ys)
