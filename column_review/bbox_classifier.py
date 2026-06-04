"""Tiny CNN that re-classifies YOLO bbox crops as column / not-column.

Two-stage architecture: YOLO + post-process produces candidates, this
classifier filters them. The detector stays frozen (no catastrophic
forgetting); only this classifier trains on user corrections.

Training data:
- positives: synthetic column tiles from `generate_column.py`,
             human-drawn FN_ADDED crops, and explicit TP confirmations.
- negatives: FP crops persisted by `scripts/hard_negative_pool.py`.

Inference data: every surviving bbox from `run_pipeline`, cropped with
the same 24 px margin convention as the hard-neg pool, resized to
64×64 grayscale, fed through the CNN. Boxes scoring < threshold are
dropped before centre-distance NMS.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import numpy as np


# Crop / input geometry constants — MUST match the values
# `scripts/hard_negative_pool.py` uses when it persists FP crops, otherwise
# train and inference see different distributions.
CROP_MARGIN_PX  = 24
CLASSIFIER_SIZE = 64


def _build_model():
    """Build the 5-layer CNN. Lazy import of torch so module import is
    cheap when the classifier isn't in use."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),                                # 32×32
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),                                # 16×16
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),                                # 8×8
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),                        # 1×1
        nn.Flatten(),
        nn.Linear(128, 1),
    )


# (path, mtime, size, device) → (model, device). Same stat-based key as the
# YOLO cache in `column_review/inference.py::_get_or_load_model`, so an
# overwrite-promote (`cp column_classifier_new.pt column_classifier.pt`)
# auto-invalidates without a server restart.
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def load_classifier(weights_path: Path | str, device: str | None = None):
    """Return a cached classifier ready for `predict_batch`.

    Stat-based cache key — promoting a freshly trained weights file by
    overwrite invalidates the cache automatically.
    """
    import torch
    p = Path(weights_path)
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size, device or "auto")
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False
        chosen = device or ("cuda:0" if cuda_ok else "cpu")
        print(f"[classifier] loading {p.name} on {chosen}…", flush=True)
        model = _build_model()
        state = torch.load(str(p), map_location=chosen, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        model.to(chosen)
        _CACHE[key] = (model, chosen)
        return model, chosen


def crop_64x64(img_gray: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    """Crop a 64×64 grayscale patch centered on `bbox`, with the same
    24 px margin the hard-neg pool uses, resized via cv2.INTER_AREA
    (anti-aliased downscale) — typical column boxes are ~16-50 px so the
    crop is always being downscaled, never upscaled.

    Public so the training script and any future caller share the EXACT
    same crop geometry as the inference path — drift here means train
    and inference see different distributions without anything flagging it.
    """
    import cv2
    H, W = img_gray.shape
    x1, y1, x2, y2 = bbox
    cx1 = max(0, int(x1) - CROP_MARGIN_PX)
    cy1 = max(0, int(y1) - CROP_MARGIN_PX)
    cx2 = min(W, int(x2) + CROP_MARGIN_PX)
    cy2 = min(H, int(y2) + CROP_MARGIN_PX)
    patch = img_gray[cy1:cy2, cx1:cx2]
    if patch.size == 0:
        return np.zeros((CLASSIFIER_SIZE, CLASSIFIER_SIZE), dtype=np.uint8)
    return cv2.resize(patch, (CLASSIFIER_SIZE, CLASSIFIER_SIZE),
                      interpolation=cv2.INTER_AREA)


def crop_batch(img_gray: np.ndarray, boxes) -> np.ndarray:
    """Build (N, 64, 64) uint8 tensor from N bboxes against `img_gray`.
    `boxes` may be any iterable of (x1, y1, x2, y2) — ndarray, list of
    lists, list of tuples all work."""
    n = len(boxes)
    if not n:
        return np.zeros((0, CLASSIFIER_SIZE, CLASSIFIER_SIZE), dtype=np.uint8)
    return np.stack([crop_64x64(img_gray, b) for b in boxes], axis=0)


def predict_batch(img_gray: np.ndarray, boxes, *,
                  weights_path: Path | str,
                  threshold: float = 0.5,
                  batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Score N boxes against the classifier. Returns (probs, keep_mask).

    `keep_mask[i] = probs[i] >= threshold`. Probabilities are sigmoid
    over the single output logit — calling code can re-threshold for
    audit / debugging without re-running inference.
    """
    import torch
    n = len(boxes)
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
    model, device = load_classifier(weights_path)
    crops = crop_batch(img_gray, boxes)        # (N, 64, 64) uint8 on CPU
    probs_out = np.zeros((n,), dtype=np.float32)
    # Keep the full CPU buffer uint8; per chunk: copy uint8 → device,
    # then cast to float32 + scale on the device. Drops the CPU float32
    # peak by 4× vs. an upfront .float().div_(255.0) on the full tensor.
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = torch.from_numpy(crops[i:i + batch_size])
            chunk = chunk.to(device, non_blocking=True).unsqueeze(1).float().div_(255.0)
            logits = model(chunk).squeeze(-1)
            probs_out[i:i + batch_size] = torch.sigmoid(logits).cpu().numpy()
    keep = probs_out >= float(threshold)
    return probs_out, keep
