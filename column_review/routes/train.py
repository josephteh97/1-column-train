"""Train-classifier + retrain-status routes.

The HITL workflow now has exactly one training loop — the CNN
classifier. POST /api/train-classifier spawns the training subprocess;
the existing /api/jobs/latest + /api/jobs/{id}/log routes keep the
status pill and log panel polling the same way they did when YOLO
retrain shared this infrastructure.
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
    start_classifier_train,
)
from column_review.routes.detections import validate_session

# Ensure scripts/ is importable so the prerequisite check (which lives
# co-located with the training script — single source of truth for
# what "training needs") can be hoisted to module top instead of paid
# per /api/train-classifier request.
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from scripts.train_bbox_classifier import check_prerequisites  # noqa: E402


router = APIRouter()


class TrainClassifierRequest(BaseModel):
    """Body for POST /api/train-classifier — no tunables.

    Classifier training is intentionally a one-click action:
    - YOLO weights are never touched (it cannot regress the detector).
    - The output `column_classifier.pt` is auto-promoted (overwritten
      each run by design) — no manual `cp` step.
    - Defaults (epochs=30, lr=1e-3) are baked into the script; the
      reviewer doesn't need to know them.
    """
    session_id: str


@router.post("/api/train-classifier")
def post_train_classifier(req: TrainClassifierRequest, request: Request):
    """Spawn `scripts/train_bbox_classifier.py` as a background job.

    Returns 412 if the preflight check (delegated to
    `scripts/train_bbox_classifier.check_prerequisites`) finds a
    missing positive or negative source. Otherwise spawns the
    subprocess and returns the new `retrain_jobs` row info so the
    status pill picks it up on the next `/api/jobs/latest` poll.
    """
    cfg = request.app.state.config
    db_path = cfg.get("db_path")
    project_root = cfg["project_root"]

    validate_session(req.session_id, db_path)

    # Preflight delegates to the script (hoisted import above) so a
    # flag rename in the script (e.g. `--canvases` → `--n`) can't
    # silently make the API response message wrong.
    missing = check_prerequisites()
    if missing:
        raise HTTPException(
            status_code=412,
            detail={
                "error":   "classifier_prerequisites_missing",
                "missing": missing,
            },
        )

    job_info = start_classifier_train(
        project_root=project_root, db_path=db_path,
    )
    return {"ok": True, "spawned": True, "retrain_job": job_info}


@router.get("/api/jobs/latest")
def get_jobs_latest(request: Request):
    """Return the most-recent `retrain_jobs` row, or `{job: None}`."""
    cfg = request.app.state.config
    job = latest_job(cfg.get("db_path"))
    return {"job": job}


@router.get("/api/jobs/{job_id}/log")
def get_jobs_log(job_id: int, request: Request,
                 tail: int = 300):
    """Return the last `tail` lines of a retrain job's tee'd log.

    Frontend polls this every 2 s while the job is non-terminal. The
    response also includes the current status so the poller can stop
    once the job reaches `completed` or `failed`.
    """
    cfg = request.app.state.config
    project_root = cfg["project_root"]
    db_path = cfg.get("db_path")
    # Look up the job's status from the DB to gate polling.
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
