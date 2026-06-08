"""Annotated-image export route.

Renders detection bboxes onto the source A0 raster at full resolution
and saves a PNG under `output/`. Server-side Pillow draw, no browser
downsampling, no re-inference.

The provenance footer reads model versions from `px_detections.json`'s
`meta.rescue_version` / `meta.classifier_version` (mtime epochs written
by `inference.py` when the cascade ran), so the footer ALWAYS matches
the bboxes — never the "footer reflects loaded weights, bboxes reflect
last inference" caveat that an in-line `column_rescue.pt` stat would
introduce.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from column_review.db import get_connection
from column_review.jobs import JOBS_DIR, resolve_source_path
from column_review.routes.detections import (
    _compute_states,
    _read_px,
    validate_session,
)


router = APIRouter()


class ExportAnnotatedRequest(BaseModel):
    job_id:     str
    drawing_id: str
    session_id: str


def _font(size: int):
    """Pick a legible TrueType font; fall back without raising."""
    from PIL import ImageFont
    for candidate in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf",
                      "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fmt_ts(t) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M",
                             time.localtime(float(t)))
    except (TypeError, ValueError):
        return "absent"


def _provenance_footer_text(project_root: Path, px_meta: dict) -> str:
    """Single-line footer: detect | rescue | classifier | exported.

    `rescue_version` / `classifier_version` come from `px_meta` —
    inference.py wrote them as `.pt` mtime epochs at the time the
    bboxes were produced. Auxiliary fields (`gate_status`,
    `best_val_acc`, `epochs_trained`) read from `.meta.json`; missing
    files degrade each field to `absent` rather than failing the
    export.
    """
    detect_path = project_root / "column_detect.pt"
    detect_mtime = (_fmt_ts(detect_path.stat().st_mtime)
                    if detect_path.is_file() else "absent")

    def _meta(fname: str) -> dict:
        try:
            return json.loads((project_root / fname).read_text())
        except (OSError, ValueError):
            return {}

    rescue = _meta("column_rescue.meta.json")
    classifier = _meta("column_classifier.meta.json")

    try:
        val_acc_s = f"{float(classifier.get('best_val_acc')):.3f}"
    except (TypeError, ValueError):
        val_acc_s = "absent"

    return (
        f"detect: column_detect.pt mtime={detect_mtime}  |  "
        f"rescue: bbox-version={_fmt_ts(px_meta.get('rescue_version'))} "
        f"gate={rescue.get('gate_status', 'absent')} "
        f"epochs={rescue.get('epochs_trained', 'absent')}  |  "
        f"classifier: bbox-version={_fmt_ts(px_meta.get('classifier_version'))} "
        f"val_acc={val_acc_s} "
        f"epochs={classifier.get('epochs_trained', 'absent')}  |  "
        f"exported: {_fmt_ts(time.time())}"
    )


def _render_annotated(src_path: Path,
                      detections: list[tuple[float, float, float,
                                              float, float]],
                      footer_text: str,
                      out_path: Path) -> None:
    """Render bboxes + provenance footer onto the source image.

    Operates entirely on the RGB image — no RGBA round-trip — so peak
    memory is ~1× the source pixel buffer (the previous overlay/paste
    path was ~2.5×). `compress_level=1` cuts PNG encode wall time
    3-5× at a ~30% file-size cost: acceptable for a stakeholder
    hand-off artifact (one click, not a hot path).
    """
    from PIL import Image, ImageDraw
    img = Image.open(src_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font = _font(28)
    stroke = (220, 20, 20)
    for x1, y1, x2, y2, score in detections:
        draw.rectangle([x1, y1, x2, y2], outline=stroke, width=4)
        draw.text((x1 + 4, y1 + 4), f"{score:.2f}",
                  fill=stroke, font=font)

    W, H = img.size
    band_h = 60
    draw.rectangle([0, H - band_h, W, H], fill=(0, 0, 0))
    draw.text((20, H - band_h + 14), footer_text,
              fill=(255, 255, 255), font=font)

    img.save(out_path, "PNG", compress_level=1)


@router.post("/api/export-annotated")
def post_export_annotated(req: ExportAnnotatedRequest, request: Request):
    """Render current detections + provenance footer to a PNG.

    Read-only with respect to `corrections.db`, model weights, and
    the absorption gate. REMOVED slots (user undid their own FN_ADDED)
    are hidden, matching the canvas overlay.

    Output: `output/<drawing_id>_annotated_<unix_ts>.png`.
    """
    cfg = request.app.state.config
    db_path = cfg.get("db_path")
    project_root: Path = cfg["project_root"]

    validate_session(req.session_id, db_path)

    px_path = JOBS_DIR / req.job_id / "px_detections.json"
    if not px_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"px_detections.json missing for job {req.job_id}",
        )
    det = _read_px(px_path)
    cols = det.get("columns", [])

    src_path = resolve_source_path(
        det=det, images_dir=cfg.get("images_dir"),
        missing_status=404, gone_status=404,
    )

    conn = get_connection(db_path)
    try:
        states = _compute_states(cols, req.job_id, conn)
    finally:
        conn.close()

    rendered: list[tuple[float, float, float, float, float]] = []
    for i, c in enumerate(cols):
        if states.get(i) == "REMOVED":
            continue
        bb = c.get("bbox") or []
        if len(bb) < 4:
            continue
        rendered.append((float(bb[0]), float(bb[1]),
                         float(bb[2]), float(bb[3]),
                         float(c.get("score", 1.0))))

    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{req.drawing_id}_annotated_{int(time.time())}.png"
    out_path = output_dir / out_name

    _render_annotated(
        src_path, rendered,
        _provenance_footer_text(project_root, det.get("meta", {})),
        out_path,
    )

    rel = out_path.relative_to(project_root)
    print(f"[export] job={req.job_id[:8]} dets={len(rendered)} → {rel}",
          flush=True)
    return {"ok": True, "path": str(rel), "n_rendered": len(rendered)}
