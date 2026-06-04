"""File picker and open-session routes.

Routes:
    GET  /api/drawings     → list drawing IDs with a built DZI pyramid
    POST /api/open         → bootstrap (or reuse) a job_id for
                              (drawing_id, reviewer_id)

Selecting a drawing whose tile pyramid is missing returns a typed 412
with the `hitl.py ingest` hint — the UI's responsibility is to surface
that diagnostic, NEVER to render a blank canvas (spec R2 + the
`tile_pyramid_missing` requirement).
"""
from __future__ import annotations

import sqlite3
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from column_review.db import get_connection
from column_review.jobs import (
    find_or_create_job,
    list_drawings,
    resolve_drawing,
)


router = APIRouter()


class OpenRequest(BaseModel):
    drawing_id: str
    reviewer_id: str


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
