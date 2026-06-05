"""Tiny load + predict helper for the rescue YOLO (`column_rescue.pt`).

The rescue YOLO is the second proposer in the two-YOLO combined
detector. It runs in parallel with the frozen `column_detect.pt`
during inference; its proposals are unioned with the main detector's
before the post-process pipeline.

Symmetry with the existing main-YOLO loader
(`column_review.inference._get_or_load_model`):
  - same stat-based cache key (path, mtime, size, device) so promoting
    a freshly trained `column_rescue.pt` by overwrite auto-invalidates
    without a server restart
  - same lazy ultralytics import so module import stays cheap when the
    cascade is running classifier-free

Soft-fail policy (REQ "Rescue model graceful degradation"): a missing
or unloadable `column_rescue.pt` returns an empty prediction list with
a one-shot stderr diagnostic. The cascade falls back to main-YOLO-only
output. This is the rollback safety net for the two-YOLO architecture.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Sequence


# Per-process cache. (path, mtime, size, device) → (model, device).
# Mirrors the YOLO main-detector cache in `column_review.inference`.
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()

# Single-shot diagnostic flags so a missing rescue weights file logs
# once per process, not once per tile.
_SOFTFAIL_LOGGED: set[str] = set()
_SOFTFAIL_LOCK = threading.Lock()


def _logged_once(key: str, msg: str) -> None:
    with _SOFTFAIL_LOCK:
        if key in _SOFTFAIL_LOGGED:
            return
        _SOFTFAIL_LOGGED.add(key)
    print(msg, file=sys.stderr, flush=True)


def load_rescue(weights_path: Path | str,
                device: str | None = None):
    """Return `(model, device)` for the rescue YOLO at `weights_path`,
    or `(None, None)` if the file is absent, unstatable, or unloadable.

    Soft-fail by design — callers MUST handle the `None` case by
    falling back to main-YOLO-only output. This keeps the cascade
    resilient when `column_rescue.pt` hasn't been produced yet
    (initial deployment) or has been deleted for rollback.
    """
    p = Path(weights_path)
    try:
        st = p.stat()
        key = (str(p), st.st_mtime, st.st_size, device or "auto")
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
            if hit is not None:
                return hit
            chosen = device
            if chosen is None:
                try:
                    import torch
                    chosen = "cuda:0" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    chosen = "cpu"
            print(f"[rescue] loading {p.name} on {chosen}…", flush=True)
            from ultralytics import YOLO
            model = YOLO(str(p))
            _CACHE[key] = (model, chosen)
            return model, chosen
    except Exception as e:
        # Absent file (FileNotFoundError), unstatable, corrupt state-dict,
        # version mismatch, half-written .pt mid-promotion — all soft-fail
        # so the cascade keeps producing main-YOLO-only output. One
        # diagnostic per (path, mtime-ish) avoids log spam.
        _logged_once(
            f"soft-fail::{p}",
            f"[rescue] {p} unavailable ({type(e).__name__}: {e}) "
            f"→ main-detector-only fallback",
        )
        return None, None


def predict_tile(model, tile_image, *, conf_threshold: float = 0.4,
                 device: str | None = None
                 ) -> list[tuple[float, float, float, float, float]]:
    """Run the rescue YOLO on ONE tile-sized image.

    Returns `[(x1, y1, x2, y2, score), ...]` in tile-local pixel
    coordinates. The caller is responsible for adding the tile origin
    when mapping to drawing-space pixels.

    `model=None` (the soft-fail case from `load_rescue`) returns an
    empty list, never raises.

    `tile_image` is whatever ultralytics' `YOLO.predict` accepts — a
    PIL.Image, a numpy array, or a path. Mirrors the main YOLO
    inference call surface so the cascade can pass the same tile
    object to both.
    """
    if model is None:
        return []
    # `verbose=False` keeps the per-tile prediction log from drowning
    # the rescue-call cadence (one call per tile, dozens per drawing).
    results = model.predict(tile_image, conf=conf_threshold,
                            verbose=False, device=device)
    if not results:
        return []
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    # ultralytics' Boxes object exposes .xyxy and .conf as tensors. The
    # cpu().tolist() pattern matches what the main-YOLO call uses
    # downstream in run_inference.
    xyxy  = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    return [(x1, y1, x2, y2, c)
            for (x1, y1, x2, y2), c in zip(xyxy, confs)]


def predict_batch(model, tile_images: Sequence,
                  *, conf_threshold: float = 0.4,
                  device: str | None = None
                  ) -> list[list[tuple[float, float, float, float, float]]]:
    """Batch wrapper around `predict_tile` for the inference call site
    that processes a list of tiles at once. Result shape parallels the
    input list."""
    if model is None or not tile_images:
        return [[] for _ in tile_images]
    return [predict_tile(model, t, conf_threshold=conf_threshold,
                         device=device) for t in tile_images]
