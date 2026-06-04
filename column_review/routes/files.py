"""File picker and open-session routes.

Routes:
    GET  /api/drawings        → list drawing IDs with a built DZI pyramid
    POST /api/open            → bootstrap a job for an ingested drawing
                                  + DZI tile source
    GET  /api/local-images    → list PNG/JPG files in `--images-dir`
                                  (the direct-image mode that bypasses
                                  the DZI tile pyramid pipeline)
    POST /api/open-local-image → bootstrap a job from a raw image file;
                                  OSD then loads it via its image mode
    GET  /raster/{job_id}     → serve the source image for a job
                                  (consumed by OSD's image tileSource)

Selecting a DZI drawing whose tile pyramid is missing returns a typed
412 with the `hitl.py ingest` hint. The local-images path skips the
tile pyramid entirely — for smaller floor plans (~60 MP) it's faster
to start with and works without the per-drawing ingest step.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from column_review.db import get_connection
from column_review.jobs import (
    JOBS_DIR,
    RAW_DRAWINGS_DIR,
    find_or_create_job,
    list_drawings,
    resolve_drawing,
)


router = APIRouter()


class OpenRequest(BaseModel):
    drawing_id: str
    reviewer_id: str


class OpenLocalImageRequest(BaseModel):
    """Open an arbitrary image file from the configured `images_dir`.

    `filename` is the basename inside the configured `images_dir` —
    full-path traversal is rejected. `reviewer_id` is the same as
    /api/open: a non-empty string used for session provenance.
    """
    filename:    str
    reviewer_id: str


class RenderAckRequest(BaseModel):
    job_id:       str
    drawing_id:   str
    open_ms:      float


@router.get("/api/drawings")
def get_drawings():
    """List drawings that have a `.dzi` tile pyramid on disk."""
    return {"drawings": list_drawings()}


@router.post("/api/open")
def post_open(req: OpenRequest, request: Request):
    """Resolve `(drawing_id, reviewer_id)` → bootstrap a session.

    Side effects:
    - Insert one row into `reviewer_sessions` (one per `POST /api/open`).
    - If no existing job matches the (drawing, raster_mtime) pair,
       a new `data/jobs/<job_id>/px_detections.json` is bootstrapped
       and a background thread starts writing `render.jpg`.
    """
    drawing_id = req.drawing_id.strip()
    reviewer_id = req.reviewer_id.strip()
    if not drawing_id or not reviewer_id:
        raise HTTPException(status_code=400,
                            detail="drawing_id and reviewer_id are required")

    try:
        raster_path, meta = resolve_drawing(drawing_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=412, detail=str(e))

    # `tile_pyramid_missing` is the R2 regression contract — without
    # the DZI the UI cannot render the raster, and the rest of the
    # workflow is meaningless. Return a typed 412 with the exact hint.
    if not meta.get("dzi_path"):
        raise HTTPException(
            status_code=412,
            detail={
                "error": "tile_pyramid_missing",
                "drawing_id": drawing_id,
                "hint": (
                    f"python3 scripts/hitl.py build-tiles {drawing_id}"
                ),
            },
        )

    job_id = find_or_create_job(drawing_id, raster_path)
    session_id = uuid.uuid4().hex

    conn = get_connection(request.app.state.config.get("db_path"))
    try:
        conn.execute(
            "INSERT INTO reviewer_sessions "
            "(session_id, reviewer_id, started_ts) VALUES (?, ?, ?)",
            (session_id, reviewer_id, time.time()),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "drawing_id":     drawing_id,
        "reviewer_id":    reviewer_id,
        "job_id":         job_id,
        "session_id":     session_id,
        "tile_source":    f"/tiles/{drawing_id}.dzi",
        "detections_url": f"/api/detections?job_id={job_id}",
    }


@router.get("/api/local-images")
def get_local_images(request: Request):
    """List PNG/JPG files in the configured `images_dir`.

    Returns `{exists, images_dir, images: [{filename, megapixels}, …]}`.
    The frontend's picker drawer shows these under a "Local images"
    section, separate from the DZI-ingested drawings.
    """
    cfg = request.app.state.config
    images_dir = cfg.get("images_dir")
    if not images_dir or not Path(images_dir).is_dir():
        return {"exists": False, "images_dir": str(images_dir) if images_dir else None,
                "images": []}
    images_dir = Path(images_dir)
    files = []
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        files.extend(images_dir.glob(f"*{ext}"))
    out = []
    for f in sorted(files):
        try:
            sz = f.stat().st_size
        except OSError:
            continue
        out.append({"filename": f.name, "size_bytes": sz})
    return {"exists": True, "images_dir": str(images_dir), "images": out}


@router.post("/api/open-local-image")
def post_open_local_image(req: OpenLocalImageRequest, request: Request):
    """Bootstrap a job for a raw image file under `images_dir`.

    Path-safety: `filename` MUST be a basename (no slashes); it's
    resolved against the configured `images_dir` and rejected if it
    escapes that root. The downstream pipeline (px_detections.json +
    render.jpg) is fully compatible with `/api/infer` and the
    retrain subprocess — the only difference from the DZI path is
    the OSD tile source URL on the response.
    """
    cfg = request.app.state.config
    images_dir = cfg.get("images_dir")
    if not images_dir or not Path(images_dir).is_dir():
        raise HTTPException(
            status_code=412,
            detail=(
                "No --images-dir configured. Restart with "
                "`column-review --images-dir <folder>`."
            ),
        )
    fn = req.filename.strip()
    if not fn or "/" in fn or fn.startswith("."):
        raise HTTPException(status_code=400,
                            detail="filename must be a basename, no slashes")
    raster_path = (Path(images_dir) / fn).resolve()
    try:
        raster_path.relative_to(Path(images_dir).resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="path traversal blocked")
    if not raster_path.is_file():
        raise HTTPException(status_code=412,
                            detail=f"image not found: {raster_path}")
    if not req.reviewer_id.strip():
        raise HTTPException(status_code=400, detail="reviewer_id required")

    # drawing_id = filename stem so all corrections + retrain artefacts
    # group under a stable identifier per source image.
    drawing_id = raster_path.stem
    job_id = find_or_create_job(drawing_id, raster_path)
    session_id = uuid.uuid4().hex

    conn = get_connection(cfg.get("db_path"))
    try:
        conn.execute(
            "INSERT INTO reviewer_sessions "
            "(session_id, reviewer_id, started_ts) VALUES (?, ?, ?)",
            (session_id, req.reviewer_id.strip(), time.time()),
        )
        conn.commit()
    finally:
        conn.close()

    # Tell the frontend to mount OSD in image-mode (no DZI tile pyramid)
    # against the new /raster/<job_id> route below.
    return {
        "drawing_id":       drawing_id,
        "reviewer_id":      req.reviewer_id.strip(),
        "job_id":           job_id,
        "session_id":       session_id,
        "tile_source":      f"/raster/{job_id}",
        "tile_source_type": "image",
        "detections_url":   f"/api/detections?job_id={job_id}",
    }


@router.get("/raster/{job_id}")
def get_raster(job_id: str, request: Request):
    """Serve the source image file for a job (OSD image-mode loads it).

    Path-safety: the persisted `meta.source` is hostile input (an old
    px_detections.json could record `/etc/passwd`, and symlinks bypass
    `is_file()`). Resolve symlinks and assert the result lives under
    either `RAW_DRAWINGS_DIR` (DZI-ingested drawings) or the configured
    `images_dir` (direct-image mode). Anything else → 403.
    """
    px_path = JOBS_DIR / job_id / "px_detections.json"
    if not px_path.is_file():
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    try:
        det = json.loads(px_path.read_text())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="px_detections.json corrupt")
    source = det.get("meta", {}).get("source")
    if not source:
        raise HTTPException(status_code=404, detail="no source recorded")

    try:
        resolved = Path(source).resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail=f"source file gone: {source}")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"source file gone: {resolved}")

    cfg = request.app.state.config
    allowed_roots = [RAW_DRAWINGS_DIR.resolve()]
    images_dir = cfg.get("images_dir")
    if images_dir:
        allowed_roots.append(Path(images_dir).resolve())
    contained = False
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            contained = True
            break
        except ValueError:
            continue
    if not contained:
        raise HTTPException(
            status_code=403,
            detail=f"source not under an allowed root: {resolved}",
        )

    return FileResponse(
        str(resolved),
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.post("/api/render-ack")
def post_render_ack(req: RenderAckRequest):
    """Record the open-to-first-render latency from the client.

    The frontend `performance.now()`s the span from `POST /api/open`
    success to OSD's `open` event + detections fetch settle, then
    POSTs the duration here. We log it for diagnosis and append to
    a per-job perf log when it exceeds the R3 3-second budget.
    """
    import time
    from column_review.jobs import JOBS_DIR
    print(f"[perf] render-ack job={req.job_id[:8]} "
          f"drawing={req.drawing_id} open_ms={req.open_ms:.0f}",
          flush=True)
    if req.open_ms > 3000.0:
        try:
            perf_log = JOBS_DIR / req.job_id / "perf.log"
            with perf_log.open("a", encoding="utf-8") as f:
                f.write(
                    f"{time.time():.3f} open drawing={req.drawing_id} "
                    f"open_ms={req.open_ms:.0f} "
                    f"(budget=3000ms)\n"
                )
        except OSError:
            pass
    return {"ok": True, "over_budget": req.open_ms > 3000.0}
