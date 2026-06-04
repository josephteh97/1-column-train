"""Save & Submit + retrain status routes.

Flow:
    1. Frontend POSTs `/api/submit` with `confirm=False`.
       Server validates `n_total >= min_corrections`, returns the
       preview payload (counts + projected command).
    2. Frontend shows a confirm modal with the preview.
    3. On confirm, frontend POSTs `/api/submit` with `confirm=True`.
       Server spawns `scripts/retrain_yolo.py` as a background
       subprocess and returns the new `retrain_jobs.id`.
    4. Frontend polls `/api/jobs/latest` for status updates.

The confirm modal is outside the correction loop — pressing
Save & Submit is not a primary correction action. R4's
"no modals in the loop" rule holds.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from column_review.db import get_connection
from column_review.retrain_jobs import (
    corrections_count,
    latest_job,
    log_tail,
    start_classifier_train,
    start_retrain,
)
from column_review.routes.detections import validate_session


router = APIRouter()


# Default retrain parameters when the frontend doesn't override them.
# Mirrors `scripts/retrain_yolo.py`'s defaults so the projected command
# matches what the subprocess actually runs.
_DEFAULT_EPOCHS = 20
_DEFAULT_MIN_CORRECTIONS = 10


class SubmitRequest(BaseModel):
    job_id:           str
    session_id:       str
    confirm:          bool = False
    epochs:           Optional[int] = None
    min_corrections:  Optional[int] = None


@router.post("/api/submit")
def post_submit(req: SubmitRequest, request: Request):
    """Two-step: preview (confirm=False) → spawn (confirm=True).

    Refuses on either step if fewer than `min_corrections` effective
    corrections exist for this job. The refusal payload includes the
    actual count so the UI can show "10 needed, you have 4" rather
    than a generic error.
    """
    cfg = request.app.state.config
    db_path = cfg.get("db_path")
    project_root = cfg["project_root"]
    epochs = int(req.epochs or _DEFAULT_EPOCHS)
    min_corrections = int(req.min_corrections or _DEFAULT_MIN_CORRECTIONS)

    # Session check is mandatory for both steps — spawning a retrain
    # without provenance would be a worse correctness hole than a
    # mark write.
    validate_session(req.session_id, db_path)

    counts = corrections_count(req.job_id, db_path)
    if counts["n_total"] < min_corrections:
        raise HTTPException(
            status_code=412,
            detail={
                "error": "min_corrections_not_met",
                "needed": min_corrections,
                "have": counts["n_total"],
                "hint": (
                    f"Mark at least {min_corrections} corrections "
                    f"before submitting. You currently have "
                    f"{counts['n_total']} effective."
                ),
            },
        )

    cmd = [
        f"python3 scripts/retrain_yolo.py",
        f"--epochs {epochs}",
        f"--min-corrections {min_corrections}",
    ]

    if not req.confirm:
        # Preview step — return what would happen.
        return {
            "ok":              True,
            "preview":         True,
            "n_fp":            counts["n_fp"],
            "n_fn_added":      counts["n_fn_added"],
            "n_total":         counts["n_total"],
            "epochs":          epochs,
            "min_corrections": min_corrections,
            "command":         " ".join(cmd),
            "projected_runtime_estimate": (
                f"~{epochs * 30}s to ~{epochs * 90}s on RTX 4000"
            ),
        }

    # Confirm=True — spawn the retrain subprocess.
    job_info = start_retrain(
        epochs=epochs,
        min_corrections=min_corrections,
        project_root=project_root,
        db_path=db_path,
    )
    return {
        "ok":          True,
        "preview":     False,
        "spawned":     True,
        "retrain_job": job_info,
        "n_fp":        counts["n_fp"],
        "n_fn_added":  counts["n_fn_added"],
    }


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

    Returns 412 with a copy-paste hint if the synthetic dataset is
    missing (`generate_column.py` hasn't run) or the hard-negative pool
    is empty (no FP corrections recorded yet). Otherwise spawns the
    subprocess and returns the same job-info shape as `/api/submit`
    so the existing retrain pill polls work unchanged.
    """
    cfg = request.app.state.config
    db_path = cfg.get("db_path")
    project_root = cfg["project_root"]

    validate_session(req.session_id, db_path)

    # Preflight delegates to the script: paths + fix-command strings
    # are owned in one place so a flag rename in the script (e.g.
    # `--canvases` → `--n`) doesn't silently make the API response
    # message wrong.
    import sys as _sys
    if str(project_root) not in _sys.path:
        _sys.path.insert(0, str(project_root))
    from scripts.train_bbox_classifier import check_prerequisites
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
