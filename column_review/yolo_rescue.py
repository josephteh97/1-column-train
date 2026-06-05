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

import os
import sys
import threading
from pathlib import Path
from typing import Sequence


# Chunk size for `predict_chunked` — bounds peak GPU memory by
# capping how many 1280×1280 tiles ultralytics batches in one
# forward pass. Default 4 matches `scripts/train_yolo_rescue.py`'s
# training batch (proven to fit on the project's 8 GB GPU);
# override via the `RESCUE_YOLO_BATCH_CHUNK` env var on larger or
# shared GPUs.
_BATCH_CHUNK_DEFAULT = 4
_BATCH_CHUNK_ENV_VAR = "RESCUE_YOLO_BATCH_CHUNK"


def get_rescue_batch_chunk() -> int:
    """Resolve the rescue YOLO batch chunk size.

    Reads the `RESCUE_YOLO_BATCH_CHUNK` env var (if set) and clamps it
    to `>= 1`. Falls back to `_BATCH_CHUNK_DEFAULT` on unset, empty,
    or unparseable values. Public so consumers that crop lazily (e.g.,
    the absorption gate) can iterate at the same chunk size without
    re-implementing the env-var policy.
    """
    raw = os.environ.get(_BATCH_CHUNK_ENV_VAR, "").strip()
    if not raw:
        return _BATCH_CHUNK_DEFAULT
    try:
        n = int(raw)
    except ValueError:
        print(f"[rescue] {_BATCH_CHUNK_ENV_VAR}={raw!r} not an integer "
              f"→ falling back to default {_BATCH_CHUNK_DEFAULT}",
              file=sys.stderr, flush=True)
        return _BATCH_CHUNK_DEFAULT
    return max(1, n)


def unpack_yolo_result(r) -> list[tuple[float, float, float, float, float]]:
    """Convert one ultralytics Results to `[(x1, y1, x2, y2, score), …]`
    in tile-local pixel coordinates.

    Shared by `predict_tile` (single result) and `predict_chunked`
    (per-result loop). The next ultralytics API drift (e.g., `.conf`
    rename) only needs to be tracked here.
    """
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy  = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    return [(x1, y1, x2, y2, c)
            for (x1, y1, x2, y2), c in zip(xyxy, confs)]


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
    return unpack_yolo_result(results[0])


def predict_chunked(model, tile_images: Sequence,
                    *, conf_threshold: float = 0.4,
                    chunk_size: int | None = None,
                    device: str | None = None,
                    ) -> list[list[tuple[float, float, float, float, float]]]:
    """Run `model.predict` on N tiles in chunks of `chunk_size`.

    Ultralytics' .predict accepts a list and batches the GPU forward
    pass — but activation memory for 1280×1280 inputs scales linearly
    with the batch, so passing all N tiles in one call can blow up a
    shared 8 GB GPU. Chunking caps peak memory at the per-chunk
    batch size.

    `chunk_size` defaults to `get_rescue_batch_chunk()`, which reads
    the `RESCUE_YOLO_BATCH_CHUNK` env var (validated, clamped to
    `>= 1`) and falls back to `_BATCH_CHUNK_DEFAULT`. Override on
    larger / freer GPUs for fewer Python+CUDA launches; lower it on
    smaller GPUs to avoid OOM.

    Returns parallel lists of `(x1, y1, x2, y2, score)` boxes in
    tile-local coordinates — `len(result) == len(tile_images)`
    invariantly. Callers add the tile origin when mapping to
    drawing-space pixels.
    """
    if model is None or not tile_images:
        return [[] for _ in tile_images]
    if chunk_size is None:
        chunk_size = get_rescue_batch_chunk()
    out: list[list[tuple[float, float, float, float, float]]] = []
    for i in range(0, len(tile_images), chunk_size):
        results = model.predict(
            list(tile_images[i:i + chunk_size]),
            conf=conf_threshold, verbose=False, device=device,
        )
        for r in results:
            out.append(unpack_yolo_result(r))
    # Parallel-list contract — defensive against any future
    # ultralytics version that drops a Result for a blank input.
    # A silent mismatch here would misalign downstream zip().
    assert len(out) == len(tile_images), (
        f"predict_chunked: ultralytics returned {len(out)} results for "
        f"{len(tile_images)} input tiles — parallel-list contract broken."
    )
    return out
