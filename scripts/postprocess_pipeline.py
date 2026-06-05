"""The post-inference pipeline as a shared module.

`test_column.ipynb` and `column_review/inference.py` both consume this
module so any tuning lands in one place. The caller is responsible for
producing the input boxes — in the two-YOLO combined detector, that
input is the union of `column_detect.pt` and `column_rescue.pt`
predictions after a cross-detector NMS pass at the union threshold
(see `column_review.inference.run_inference` for the union step).

Filters in order:
  (1) ASPECT      — drop max(w,h)/min(w,h) > MAX_ASPECT.
  (2) SIZE        — drop sides outside [MIN_SIDE_PX, MAX_SIDE_PX].
  (3) SHAPE       — require fill_ratio >= MIN_FILL_RATIO OR
                     border-ring dark_ratio >= MIN_BORDER_RATIO.
  (3.5) OCR-TEXT  — drop bboxes where Tesseract reads >= OCR_MIN_CHARS
                     alphanumeric chars IN A SINGLE token at conf >=
                     OCR_MIN_CONF. (Per-token, not summed across tokens
                     — fixes a false-FN on legit columns whose interior
                     contains two stray single-char readings.)
  (4) CENTRE-NMS  — drop within CENTRE_DIST_PX of a higher-conf
                     detection.
  (5) IoU-NMS     — backup at NMS_IOU_BACKUP for partial overlaps.

The previous CNN-classifier veto stage (between OCR and centre-NMS)
was removed when the two-YOLO architecture replaced it — the rescue
YOLO's missing-label training at FP locations performs that role
end-to-end.

(A prior stair-mask pre-filter using HoughLinesP was dropped during
an earlier refactor — it over-fired on real plans, masking the whole
drawing. Re-add as a dedicated module if a stair-FP need recurs.)

OOD hard-fail: `run_pipeline(..., input_dpi=300, tile_detection_counts=...)`
runs OOD before any filter. Without `tile_detection_counts`, only the
DPI check fires and an audit note records the loss of spread info.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ── Defaults — overridable via PostprocessConfig ──
GREY_DARK_THRESH = 200
MIN_FILL_RATIO   = 0.40
MIN_BORDER_RATIO = 0.35
BORDER_THICK_PX  = 2
MAX_ASPECT       = 2.0
MIN_SIDE_PX      = 12
MAX_SIDE_PX      = 60
CENTRE_DIST_PX   = 50
NMS_IOU_BACKUP   = 0.15

OCR_CROP_PAD_PX  = 4
OCR_MIN_CONF     = 50
OCR_MIN_CHARS    = 2
OCR_CHAR_WHITELIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz0123456789-"
)


@dataclass
class PostprocessConfig:
    grey_dark_thresh: int   = GREY_DARK_THRESH
    min_fill_ratio:   float = MIN_FILL_RATIO
    min_border_ratio: float = MIN_BORDER_RATIO
    border_thick_px:  int   = BORDER_THICK_PX
    max_aspect:       float = MAX_ASPECT
    min_side_px:      int   = MIN_SIDE_PX
    max_side_px:      int   = MAX_SIDE_PX
    centre_dist_px:   int   = CENTRE_DIST_PX
    nms_iou_backup:   float = NMS_IOU_BACKUP
    ocr_crop_pad_px:  int   = OCR_CROP_PAD_PX
    ocr_min_conf:     int   = OCR_MIN_CONF
    ocr_min_chars:    int   = OCR_MIN_CHARS
    ocr_char_whitelist: str = OCR_CHAR_WHITELIST
    use_ocr_filter:   bool  = True
    # CNN classifier veto stage — Architecture C's FP-rejection
    # specialist. Off by default; opt in by setting
    # use_classifier_filter=True and classifier_weights to a trained
    # column_classifier.pt. See column_review/bbox_classifier.py for
    # the model + scripts/train_bbox_classifier.py for the training CLI.
    use_classifier_filter: bool  = False
    classifier_weights:    str   = ""
    classifier_threshold:  float = 0.5


DEFAULT_CONFIG = PostprocessConfig()


@dataclass
class AuditLog:
    raw:               int = 0
    after_aspect:      int = 0
    after_size:        int = 0
    after_shape:       int = 0
    after_ocr:         int | None = None
    after_classifier:  int | None = None
    after_centre_nms:  int = 0
    final:             int = 0
    notes:             list[str] = field(default_factory=list)


def _bbox_aspect(b):
    w = max(1.0, b[2] - b[0])
    h = max(1.0, b[3] - b[1])
    return max(w, h) / min(w, h)


def _bbox_side_range(b):
    w = b[2] - b[0]
    h = b[3] - b[1]
    return min(w, h), max(w, h)


def _shape_passes(b, img_gray, cfg: PostprocessConfig):
    ix1, iy1 = max(0, int(b[0])), max(0, int(b[1]))
    ix2, iy2 = min(img_gray.shape[1], int(b[2])), min(img_gray.shape[0], int(b[3]))
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return False
    patch = img_gray[iy1:iy2, ix1:ix2]
    dark = patch <= cfg.grey_dark_thresh
    fill_ratio = float(dark.sum()) / dark.size
    if fill_ratio >= cfg.min_fill_ratio:
        return True
    h, w = patch.shape
    bt = min(cfg.border_thick_px, min(h, w) // 2)
    border_mask = np.zeros_like(dark)
    border_mask[:bt, :] = True
    border_mask[-bt:, :] = True
    border_mask[:, :bt] = True
    border_mask[:, -bt:] = True
    border_total = border_mask.sum()
    border_dark = (dark & border_mask).sum()
    return float(border_dark) / max(1, border_total) >= cfg.min_border_ratio


def _bbox_has_text(b, img_gray, pytesseract, cfg: PostprocessConfig) -> bool:
    """True if Tesseract reads >= cfg.ocr_min_chars alphanumeric chars
    IN A SINGLE TOKEN at conf >= cfg.ocr_min_conf inside the bbox crop.

    Per-token, not summed across tokens — this avoids false positives
    where two single-char garbage tokens add up to the threshold.
    """
    pad = cfg.ocr_crop_pad_px
    ix1 = max(0, int(b[0]) - pad)
    iy1 = max(0, int(b[1]) - pad)
    ix2 = min(img_gray.shape[1], int(b[2]) + pad)
    iy2 = min(img_gray.shape[0], int(b[3]) + pad)
    if ix2 - ix1 < 6 or iy2 - iy1 < 6:
        return False
    crop = img_gray[iy1:iy2, ix1:ix2]
    config = (
        f"--oem 3 --psm 6 "
        f"-c tessedit_char_whitelist={cfg.ocr_char_whitelist}"
    )
    try:
        data = pytesseract.image_to_data(
            crop, config=config, output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return False
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        if not txt:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < cfg.ocr_min_conf:
            continue
        alnum = sum(1 for ch in txt if ch.isalnum())
        if alnum >= cfg.ocr_min_chars:
            return True
    return False


def _centre_dist_nms(boxes, scores, max_d):
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    kept_idx, kept_c = [], []
    for i in order:
        b = boxes[i]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        if not any(math.hypot(cx - kx, cy - ky) <= max_d for kx, ky in kept_c):
            kept_idx.append(i)
            kept_c.append((cx, cy))
    return kept_idx


def run_pipeline(
    img_gray: np.ndarray,
    boxes: Sequence,
    scores: Sequence,
    *,
    config: PostprocessConfig = DEFAULT_CONFIG,
    input_dpi: int | None = None,
    tile_detection_counts: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, AuditLog]:
    """Run the 7-filter post-inference pipeline.

    Parameters
    ----------
    img_gray              : 2-D uint8 grayscale source plan.
    boxes                 : iterable of (x1, y1, x2, y2) in global image coords.
    scores                : iterable of float confidences parallel to boxes.
    config                : PostprocessConfig overrides; defaults are recommended.
    input_dpi             : if not None, runs OOD hard-fail before any filter.
    tile_detection_counts : per-tile raw detection counts from the tiling
                            loop. Required when input_dpi is set so the OOD
                            ceiling can detect single-tile detection storms.
                            If omitted, OOD falls back to a global mean check
                            and notes the loss of spread information.

    Returns
    -------
    boxes_final, scores_final, audit
    """
    # Hard invariant — must survive `python -O` (which strips bare asserts).
    # An RGB array silently corrupts _shape_passes because min(h, w) collapses
    # to the 3-channel dim; raise instead so the failure is loud and reliable.
    # Guard isinstance FIRST so a None / PIL.Image / list / cv2.imread-failure
    # input gets a clean TypeError naming the type, not an AttributeError on
    # the subsequent .ndim access.
    if not isinstance(img_gray, np.ndarray):
        raise TypeError(
            f"img_gray must be np.ndarray, got {type(img_gray).__name__}"
        )
    if img_gray.ndim != 2 or img_gray.dtype != np.uint8:
        raise TypeError(
            f"img_gray must be 2-D uint8, got shape={img_gray.shape} "
            f"dtype={img_gray.dtype}"
        )

    audit = AuditLog()

    # OOD pre-check — fail loud, no silent fallback.
    if input_dpi is not None:
        # Bare sibling import works under CLI (`python3 scripts/foo.py`
        # puts scripts/ on sys.path); the column-review server imports
        # this module as `scripts.postprocess_pipeline`, so scripts/ is
        # NOT on sys.path and the bare import fails. Fall back to the
        # fully-qualified package path in that case.
        try:
            from ood_detector import check_dpi, check_tile_detections
        except ModuleNotFoundError:
            from scripts.ood_detector import (
                check_dpi, check_tile_detections,
            )
        check_dpi(input_dpi)
        if tile_detection_counts is None:
            # Per-tile spread check is unavailable without real per-tile
            # counts. We deliberately do NOT fake them by spreading the
            # global mean N times — that would defeat the storm-detection
            # ceiling. Caller should pass tile_detection_counts from
            # `scripts/tiled_inference.tiled_predict` to enable it.
            audit.notes.append(
                "tile-detection spread check skipped (no tile_detection_counts)"
            )
        else:
            check_tile_detections(tile_detection_counts)

    # Filter pipeline
    if len(boxes) == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        audit.final = 0
        audit.notes.append("empty raw input — pipeline short-circuited")
        return empty, np.zeros((0,), dtype=np.float32), audit

    boxes_arr  = np.array(list(boxes),  dtype=np.float32)
    scores_arr = np.array(list(scores), dtype=np.float32)
    audit.raw = len(boxes_arr)

    # (1) Aspect
    mask = np.array([_bbox_aspect(b) <= config.max_aspect for b in boxes_arr])
    boxes_arr = boxes_arr[mask]; scores_arr = scores_arr[mask]
    audit.after_aspect = len(boxes_arr)

    # (2) Size
    def _sz_ok(b):
        mn, mx = _bbox_side_range(b)
        return config.min_side_px <= mn and mx <= config.max_side_px
    mask = np.array([_sz_ok(b) for b in boxes_arr]) if len(boxes_arr) else np.zeros(0, dtype=bool)
    boxes_arr = boxes_arr[mask]; scores_arr = scores_arr[mask]
    audit.after_size = len(boxes_arr)

    # (3) Shape
    mask = np.array([_shape_passes(b, img_gray, config) for b in boxes_arr]) \
        if len(boxes_arr) else np.zeros(0, dtype=bool)
    boxes_arr = boxes_arr[mask]; scores_arr = scores_arr[mask]
    audit.after_shape = len(boxes_arr)

    # (3.5) OCR-text
    if config.use_ocr_filter and len(boxes_arr):
        try:
            import pytesseract  # noqa
            _OCR_OK = True
        except ImportError:
            _OCR_OK = False
            audit.notes.append("OCR filter skipped: pytesseract not installed")
        if _OCR_OK:
            text_mask = np.array([
                _bbox_has_text(b, img_gray, pytesseract, config) for b in boxes_arr
            ])
            boxes_arr  = boxes_arr [~text_mask]
            scores_arr = scores_arr[~text_mask]
            audit.after_ocr = len(boxes_arr)

    # (3.7) CNN classifier veto — Architecture C's FP-rejection
    # specialist. Runs AFTER content-aware shape/OCR filters and
    # BEFORE centre-NMS so duplicate FPs of the same wrong thing
    # don't both survive. Soft-fails if weights are missing — the
    # pipeline still produces YOLO+rescue output, never raises.
    if config.use_classifier_filter and config.classifier_weights and len(boxes_arr):
        try:
            from column_review.bbox_classifier import predict_batch
            _, keep = predict_batch(
                img_gray, boxes_arr,
                weights_path=config.classifier_weights,
                threshold=config.classifier_threshold,
            )
            dropped = int((~keep).sum())
            boxes_arr  = boxes_arr [keep]
            scores_arr = scores_arr[keep]
            audit.after_classifier = len(boxes_arr)
            if dropped:
                audit.notes.append(
                    f"classifier dropped {dropped} "
                    f"(threshold={config.classifier_threshold})"
                )
        except Exception as e:
            # Soft-fail on EVERY classifier failure mode: missing file
            # (FileNotFoundError/OSError), missing import (ImportError),
            # 0-byte/corrupt .pt or arch-mismatch state_dict
            # (RuntimeError), torch.load key errors, CUDA OOM, etc.
            # The pipeline still produces YOLO+rescue output; the
            # type name in the audit note tells operators which class
            # of failure occurred without a 500 response.
            audit.notes.append(
                f"classifier filter skipped: {type(e).__name__}: {e}"
            )

    # (4) Centre-distance NMS
    if len(boxes_arr):
        idx = _centre_dist_nms(boxes_arr.tolist(), scores_arr.tolist(), config.centre_dist_px)
        boxes_arr = boxes_arr[idx]; scores_arr = scores_arr[idx]
    audit.after_centre_nms = len(boxes_arr)

    # (5) IoU NMS backup
    if len(boxes_arr) == 0:
        boxes_final  = np.zeros((0, 4), dtype=np.float32)
        scores_final = np.zeros((0,),    dtype=np.float32)
    else:
        import torch
        import torchvision.ops as tvops
        boxes_t  = torch.tensor(boxes_arr,  dtype=torch.float32)
        scores_t = torch.tensor(scores_arr, dtype=torch.float32)
        keep = tvops.nms(boxes_t, scores_t, iou_threshold=config.nms_iou_backup)
        boxes_final  = boxes_t[keep].numpy()
        scores_final = scores_t[keep].numpy()
    audit.final = len(boxes_final)

    return boxes_final, scores_final, audit


def format_audit(audit: AuditLog) -> str:
    lines = [
        f"raw                   : {audit.raw}",
        f"after aspect          : {audit.after_aspect}",
        f"after size            : {audit.after_size}",
        f"after shape           : {audit.after_shape}",
    ]
    if audit.after_ocr is not None:
        lines.append(f"after OCR text        : {audit.after_ocr}")
    if audit.after_classifier is not None:
        lines.append(f"after CNN classifier  : {audit.after_classifier}")
    lines.extend([
        f"after centre-NMS      : {audit.after_centre_nms}",
        f"FINAL                 : {audit.final}",
    ])
    if audit.notes:
        for n in audit.notes:
            lines.append(f"  (note) {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    img = np.full((1280, 1280), 255, dtype=np.uint8)
    img[100:120, 100:120] = 0   # one filled "column"
    boxes = np.array([[98, 98, 122, 122]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    b, s, a = run_pipeline(img, boxes, scores)
    print(format_audit(a))
    print(f"out: {b.tolist()}")
