"""Train-rescue + retrain-status routes.

The HITL workflow has exactly one training loop — the rescue YOLO
(`column_rescue.pt`, yolo11n). `POST /api/train-rescue` spawns the
training subprocess; `/api/jobs/latest` + `/api/jobs/{id}/log` keep
the status pill and log panel polling.

The frozen baseline `column_detect.pt` is NEVER touched by anything
spawned from here — the rescue training script writes to a quarantine
path, the absorption gate decides whether to promote to
`column_rescue.pt`, and the main detector is out of reach by
design.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from column_review.db import get_connection
from column_review.retrain_jobs import (
    latest_job,
    log_tail,
    start_rescue_train,
)
from column_review.routes.detections import validate_session

# Ensure scripts/ is importable so the prerequisite check (which lives
# co-located with the training script — single source of truth for
# what "training needs") can be hoisted to module top instead of paid
# per request.
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from scripts.train_yolo_rescue import check_prerequisites  # noqa: E402


router = APIRouter()


class TrainRescueRequest(BaseModel):
    """Body for POST /api/train-rescue — no tunables.

    Rescue training is intentionally a one-click action:
    - `column_detect.pt` is never touched (frozen forever).
    - The output `column_rescue.pt` is promoted only by the absorption
      gate — failed gate → quarantine retained, canonical path
      unchanged, UI surfaces the diagnostic.
    - Defaults (epochs=30, batch=4, lr=5e-4) are baked into the
      script; the reviewer doesn't need to know them.
    """
    session_id: str


@router.post("/api/train-rescue")
def post_train_rescue(req: TrainRescueRequest, request: Request):
    """Spawn `scripts/train_yolo_rescue.py` as a background job.

    Returns 412 if the preflight (delegated to
    `scripts/train_yolo_rescue.check_prerequisites`) finds no training
    data. Otherwise spawns the subprocess and returns the new
    `retrain_jobs` row info so the status pill picks it up on the
    next `/api/jobs/latest` poll.
    """
    cfg = request.app.state.config
    db_path = cfg.get("db_path")
    project_root = cfg["project_root"]

    validate_session(req.session_id, db_path)

    missing = check_prerequisites()
    if missing:
        raise HTTPException(
            status_code=412,
            detail={
                "error":   "rescue_prerequisites_missing",
                "missing": missing,
            },
        )

    job_info = start_rescue_train(
        project_root=project_root, db_path=db_path,
    )
    return {"ok": True, "spawned": True, "retrain_job": job_info}


@router.get("/api/jobs/latest")
def get_jobs_latest(request: Request):
    """Return the most-recent `retrain_jobs` row, or `{job: None}`.

    The frontend's status-pill poller hits this every few seconds
    while a job is non-terminal. The response shape is unchanged from
    the pre-rescue era so the polling logic continues to work.
    """
    cfg = request.app.state.config
    job = latest_job(cfg.get("db_path"))
    return {"job": job}


@router.get("/api/jobs/{job_id}/log")
def get_jobs_log(job_id: int, request: Request,
                 tail: int = 300):
    """Return the last `tail` lines of a retrain job's tee'd log."""
    cfg = request.app.state.config
    project_root = cfg["project_root"]
    db_path = cfg.get("db_path")
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT status, started_ts, finished_ts FROM retrain_jobs "
            "WHERE id = ?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return {"job_id": job_id, "status": "unknown",
                "log": "(no such job)"}
    status = row[0]
    body = log_tail(job_id, project_root, n_lines=int(tail))
    return {"job_id": job_id, "status": status,
            "started_ts": row[1], "finished_ts": row[2],
            "log": body or "(no log yet)"}
