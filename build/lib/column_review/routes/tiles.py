"""DZI tile-pyramid serving.

Routes:
    GET /tiles/{drawing_id}.dzi                  → DZI XML manifest
    GET /tiles/{drawing_id}_files/{level}/{tile} → 256x256 JPEG tile

Path traversal is blocked by resolving against `RAW_DRAWINGS_DIR` and
asserting the resolved path stays inside it. A missing pyramid returns
a typed JSON error with the exact `hitl.py ingest` hint — never a
silent 404 page.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from column_review.jobs import RAW_DRAWINGS_DIR


router = APIRouter()


def _serve_from_raw(path_under_raw: str) -> FileResponse:
    target = (RAW_DRAWINGS_DIR / path_under_raw).resolve()
    try:
        target.relative_to(RAW_DRAWINGS_DIR.resolve())
    except ValueError:
        # Path-traversal attempt; hide existence with a 404.
        raise HTTPException(status_code=404)
    if not target.is_file():
        raise HTTPException(status_code=404)
    # Tiles are immutable on disk — long-cache them at the browser.
    return FileResponse(
        str(target),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/tiles/{drawing_id}.dzi")
def get_dzi(drawing_id: str):
    target = RAW_DRAWINGS_DIR / f"{drawing_id}.dzi"
    if not target.is_file():
        return JSONResponse(
            status_code=412,
            content={
                "error": "tile_pyramid_missing",
                "drawing_id": drawing_id,
                "hint": (
                    f"python3 scripts/hitl.py ingest <plan> "
                    f"--drawing-id {drawing_id}"
                ),
            },
        )
    return _serve_from_raw(f"{drawing_id}.dzi")


@router.get("/tiles/{drawing_id}_files/{level}/{tile_filename}")
def get_tile(drawing_id: str, level: str, tile_filename: str):
    # level is a string of digits; tile_filename is `<col>_<row>.jpg`.
    # Path-traversal guards run in `_serve_from_raw`.
    return _serve_from_raw(
        f"{drawing_id}_files/{level}/{tile_filename}"
    )
