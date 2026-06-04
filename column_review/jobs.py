"""Per-drawing job lookup and bootstrap.

`data/jobs/<job_id>/` holds the per-reviewer-session artefacts:

    px_detections.json   — the model output + any human_added entries
    render.jpg           — JPEG snapshot of the raster, consumed by
                            retrain_yolo.py's hard-negative pool

`find_or_create_job` matches existing jobs by `(source path,
raster_mtime)` so a re-ingest of the same drawing (same path on disk,
fresh pixels) does NOT inherit the stale render.jpg of the previous
job. Otherwise the human's bbox edits would silently misalign with
the new raster.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import HTTPException
from PIL import Image

from column_review.db import new_job_id


# Resolve the data tree relative to the project root via this module's
# location: <project>/column_review/jobs.py → <project>/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"
JOBS_DIR = _DATA_ROOT / "jobs"
RAW_DRAWINGS_DIR = _DATA_ROOT / "raw" / "drawings"


def _is_under(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_source_path(
    *,
    det: dict | None = None,
    px_path: Path | None = None,
    images_dir: Path | str | None = None,
    missing_status: int = 404,
    gone_status: int = 404,
) -> Path:
    """Return a symlink-resolved, allowed-roots-checked Path to a job's
    raster source. Used by `/raster/{job_id}` and `/api/infer` so both
    see the same pixel buffer — and apply the same path-safety guards.

    Pass `det` if you've already read `px_detections.json` (e.g., under
    a write lock). Otherwise pass `px_path` and the helper reads it.

    `meta.source` is hostile input (a doctored px_detections.json could
    record `/etc/passwd`, and symlinks bypass `is_file()`). The check
    enforces: source must exist, must resolve under `RAW_DRAWINGS_DIR`
    or (if supplied) `images_dir`. 403 otherwise.
    """
    if det is None and px_path is None:
        raise ValueError("either det or px_path is required")
    if det is None:
        if not px_path.is_file():
            raise HTTPException(
                status_code=missing_status,
                detail=f"job not found: {px_path.parent.name}",
            )
        try:
            det = json.loads(px_path.read_text())
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500, detail="px_detections.json corrupt",
            )
    source = det.get("meta", {}).get("source")
    if not source:
        raise HTTPException(
            status_code=missing_status,
            detail=(
                "px_detections.json missing meta.source — close the "
                "drawing and re-open it from the picker so the job is "
                "re-bootstrapped against the correct raster."
            ),
        )
    try:
        resolved = Path(source).resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(
            status_code=gone_status,
            detail=f"source file gone: {source}",
        )
    if not resolved.is_file():
        raise HTTPException(
            status_code=gone_status,
            detail=f"source file gone: {resolved}",
        )
    allowed_roots = [RAW_DRAWINGS_DIR.resolve()]
    if images_dir:
        allowed_roots.append(Path(images_dir).resolve())
    if not any(_is_under(resolved, r) for r in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail=f"source not under an allowed root: {resolved}",
        )
    return resolved


def list_drawings() -> list[str]:
    """Return all drawing IDs that have a `.dzi` tile pyramid on disk.

    Sorted alphabetically. Drawings without a DZI are excluded — the
    UI can't render them, so listing them would only invite a
    blank-canvas failure.
    """
    if not RAW_DRAWINGS_DIR.exists():
        return []
    return sorted(
        p.stem for p in RAW_DRAWINGS_DIR.glob("*.dzi")
    )


def resolve_drawing(drawing_id: str) -> tuple[Path, dict]:
    """Resolve `(raster_path, meta)` for a drawing_id via meta.json.

    Mirrors `scripts.ingest_drawings.resolve_drawing` with one tweak:
    the returned `meta` dict is enriched with `dzi_path` (the absolute
    path to `<drawing_id>.dzi`) which the file-picker route consumes.
    Raises FileNotFoundError if the drawing has not been ingested.
    """
    meta_path = RAW_DRAWINGS_DIR / f"{drawing_id}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No ingest record for drawing_id={drawing_id!r}. Run:\n"
            f"  python3 scripts/hitl.py ingest <plan> "
            f"--drawing-id {drawing_id}"
        )
    meta = json.loads(meta_path.read_text())
    raster_path = _DATA_ROOT / meta["ingested_as"]
    if not raster_path.exists():
        siblings = [p for p in RAW_DRAWINGS_DIR.glob(f"{drawing_id}.*")
                    if not p.name.endswith(".meta.json")
                    and p.suffix != ".dzi"]
        if not siblings:
            raise FileNotFoundError(
                f"meta.json points at {raster_path} but no raster "
                f"found. Re-run: python3 scripts/hitl.py ingest "
                f"<plan> --drawing-id {drawing_id}"
            )
        raster_path = max(siblings, key=lambda p: p.stat().st_mtime)
    dzi_path = RAW_DRAWINGS_DIR / f"{drawing_id}.dzi"
    meta["dzi_path"] = str(dzi_path) if dzi_path.exists() else None
    return raster_path, meta


def _bootstrap_empty_job(job_id: str, source_path: str,
                         raster_mtime: float) -> None:
    """Write a minimal `px_detections.json` for a fresh review session.

    No render.jpg here — that's the background thread (see below). The
    reviewer never reads render.jpg directly; it's consumed by
    `scripts/retrain_yolo.py` at retrain time.
    """
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    detections = {
        "columns": [],
        "meta": {
            "source": source_path,
            "created_ts": time.time(),
            "raster_mtime": raster_mtime,
            "n": 0,
        },
    }
    (job_dir / "px_detections.json").write_text(
        json.dumps(detections, indent=2))


def _spawn_render_jpg_write(job_id: str, raster_path: Path) -> None:
    """Background JPEG-write so create_app/open isn't blocked by the
    ~10–30 s encode on a 140 MP A0 raster. Idempotent."""
    def _do() -> None:
        Image.MAX_IMAGE_PIXELS = None
        render_path = JOBS_DIR / job_id / "render.jpg"
        if render_path.exists():
            return
        try:
            with Image.open(raster_path) as src:
                src.convert("RGB").save(render_path, quality=92)
        except Exception:
            import traceback
            traceback.print_exc()
    threading.Thread(target=_do, daemon=True).start()


def find_or_create_job(drawing_id: str, raster_path: Path) -> str:
    """Return an existing job ID for `(drawing_id, raster_path)` if one
    matches by both `source` AND `raster_mtime`; otherwise bootstrap a
    fresh job. The (source, raster_mtime) match guarantees
    `scripts/retrain_yolo.py` and the hard-negative pool find a
    `render.jpg` whose pixels match the bboxes saved in
    `px_detections.json`.
    """
    source_path = str(raster_path.resolve())
    raster_mtime = raster_path.stat().st_mtime
    if JOBS_DIR.exists():
        candidates = sorted(JOBS_DIR.glob("*/px_detections.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            try:
                det = json.loads(cand.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            meta = det.get("meta", {})
            if meta.get("source") != source_path:
                continue
            stored_mtime = meta.get("raster_mtime")
            if stored_mtime is None:
                meta["raster_mtime"] = raster_mtime
                det["meta"] = meta
                import os
                tmp = cand.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(det, indent=2))
                os.replace(tmp, cand)
                return cand.parent.name
            if stored_mtime == raster_mtime:
                return cand.parent.name

    job_id = new_job_id()
    _bootstrap_empty_job(job_id, source_path, raster_mtime)
    _spawn_render_jpg_write(job_id, raster_path)
    return job_id
