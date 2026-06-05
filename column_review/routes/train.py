"""Train-both + retrain-status routes (Architecture C).

The HITL workflow has ONE button → ONE endpoint → TWO trainable models.
`POST /api/train-both` spawns `scripts/train_both.py`, which runs
the CNN classifier (~30 s) then the rescue YOLO (~20 min) sequentially.
`/api/jobs/latest` + `/api/jobs/{id}/log` keep the status pill and log
panel polling.

The frozen baseline `column_detect.pt` is NEVER touched. CNN promotion
is automatic (small model, fast retrain). Rescue promotion is gated by
`scripts/absorption_gate.py` — failed gate → quarantine retained,
`column_rescue.pt` unchanged, UI surfaces the diagnostic.
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
    start_both_train,
)
from column_review.routes.detections import validate_session

# Ensure scripts/ is importable so the prerequisite checks (which live
# co-located with each training script — single source of truth for
# what "training needs") can be hoisted to module top instead of paid
# per request.
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from scripts.train_both import check_prerequisites   # noqa: E402


router = APIRouter()


class TrainBothRequest(BaseModel):
    """Body for POST /api/train-both — no tunables.

    Train-Both is intentionally a one-click action:
    - `column_detect.pt` is never touched (frozen forever).
    - CNN classifier promotes automatically on training success
      (the small model + ~30 s retrain make this safe).
    - Rescue YOLO promotes only via the absorption gate.
    - Defaults are baked into each script; the reviewer doesn't
      tune anything from the UI.
    """
    session_id: str


@router.post("/api/train-both")
def post_train_both(req: TrainBothRequest, request: Request):
    """Spawn `scripts/train_both.py` (sequential CNN → rescue) as a
    background job.

    Returns 412 if EITHER preflight check fails — Architecture C
    semantics require both models to absorb every correction in one
    cycle. The 412 payload lists every missing prereq from both
    scripts so the UI surfaces them together.
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
                "error":   "train_both_prerequisites_missing",
                "missing": missing,
            },
        )

    job_info = start_both_train(
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
