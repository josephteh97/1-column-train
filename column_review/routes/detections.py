"""Detections, marks, undo/redo, and inference routes.

Routes:
    GET  /api/detections?job_id=X → per-detection list with state
    POST /api/infer               → run YOLO on a drawing's raster,
                                      write px_detections.json
    POST /api/marks               → apply ONE mark
                                      ({FP_TOGGLE, FN_ADDED})
    POST /api/undo                → pop the latest mark for this job,
                                      apply its inverse, push to redo
    POST /api/redo                → pop from redo, re-apply, push to
                                      undo

The marks encoding mirrors the on-disk shape that
`scripts/retrain_yolo.py` already consumes:

  FP   → `corrections` row with `is_delete=1`, positive
         `element_index` matching the model detection
  FN   → APPEND `{"bbox": [...], "source": "human_added"}` to
         `data/jobs/<job_id>/px_detections.json["columns"]`,
         then write a `corrections` row with `is_delete=0` at
         the newly-assigned positive `element_index`.

Any encoding that does NOT append to the JSON would be silently
dropped by `retrain_yolo.py:432`, which reads bboxes only from the
JSON and uses corrections as a delete-set.

Concurrency: a single `threading.RLock` per process serialises the
read-modify-write cycle across `/api/infer`, `/api/marks`, `/api/undo`,
`/api/redo`, and `GET /api/detections` so a reader never observes a
torn intermediate state. The lock is per-process because the CLI is
single-server; multi-writer scenarios are out of scope for the spec.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from column_review.db import get_connection, iter_effective_corrections
from column_review.jobs import JOBS_DIR, resolve_drawing


router = APIRouter()


# Re-entrant per-process write lock. RLock because `_apply_mark` calls
# `_compute_states` while already holding the lock; a plain Lock would
# deadlock on the inner acquire.
_JOB_LOCK = threading.RLock()

# Per-job undo/redo stacks, capped at 100 entries per the R9 spec.
# Each stack entry is `(inverse_action_dict, original_action_dict)` so
# undo can apply the inverse and push the original onto the redo stack.
_UNDO: dict[str, deque] = {}
_REDO: dict[str, deque] = {}
_STACK_MAX = 100


# ──────────────────────────────────────────────────────────────────────
# Mark application primitives.
# ──────────────────────────────────────────────────────────────────────


def _read_px(px_path: Path) -> dict:
    if not px_path.exists():
        return {"columns": [], "meta": {}}
    try:
        return json.loads(px_path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"px_detections.json corrupt at {px_path}: {e}",
        )


def _write_px(px_path: Path, det: dict) -> None:
    """Atomic temp-file replace so a crash mid-write can never leave a
    truncated JSON that the next launch silently loses entries from."""
    tmp = px_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(det, indent=2))
    os.replace(tmp, px_path)


def _compute_states(cols: list, job_id: str,
                    conn: sqlite3.Connection) -> dict[int, str]:
    """Return `{element_index: state}` for the current cols list.

    State precedence:
      "FP" overrides "FN_ADDED" (the audit trail of an FP mark on a
                                 human_added slot stands)
      "FN_ADDED" overrides "UNREVIEWED" (human-drawn boxes are seen
                                          as added until marked)
      "UNREVIEWED" is the default for model detections
    `iter_effective_corrections` does the rescind-on-read so an undone
    FP mark is filtered out automatically.
    """
    state: dict[int, str] = {}
    for i, c in enumerate(cols):
        state[i] = ("FN_ADDED" if c.get("source") == "human_added"
                    else "UNREVIEWED")
    for row in iter_effective_corrections(conn, job_id=job_id):
        _job, _et, idx, _orig, _changes, is_delete, _ts = row
        if is_delete:
            state[idx] = "FP"
    return state


def _push_undo(job_id: str, original: dict, inverse: dict) -> None:
    """Record one applied mark for later undo.

    `original` is the action the user just performed; `inverse` is the
    action that would undo it. Bounded LIFO — oldest entries drop off
    the bottom of the deque past the spec's 100-level guarantee.
    """
    stack = _UNDO.setdefault(job_id, deque(maxlen=_STACK_MAX))
    stack.append((original, inverse))
    # A new mark invalidates any pending redo stack — the user has
    # branched away from the previously-undone history.
    _REDO.pop(job_id, None)


def _apply_mark_locked(job_id: str, action: dict,
                       conn: sqlite3.Connection,
                       px_path: Path, det: dict,
                       record_undo: bool) -> tuple[dict, dict[int, str]]:
    """Apply ONE action to the DB / JSON and return `(inverse, states)`.

    `det` is the parsed `px_detections.json` dict; `det["columns"]` is
    mutated in place by FN_ADDED. Passing the whole dict in (rather
    than re-reading inside) preserves the metadata (`meta.n`, etc.)
    across writes and avoids a second JSON parse per mark.

    `record_undo=False` is used by undo/redo so we don't recursively
    push entries onto the same stack we're popping from.
    """
    cols = det.setdefault("columns", [])
    kind = action.get("action")
    inverse: Optional[dict] = None

    if kind == "FP":
        idx = int(action["element_index"])
        if not (0 <= idx < len(cols)):
            raise HTTPException(status_code=400,
                                detail=f"element_index {idx} out of range")
        original = json.dumps(cols[idx])
        conn.execute(
            "INSERT OR IGNORE INTO corrections "
            "(job_id, element_type, element_index, "
            " original_element, changes, is_delete) "
            "VALUES (?, 'column', ?, ?, '{}', 1)",
            (job_id, idx, original),
        )
        inverse = {"action": "RESCIND_FP", "element_index": idx}

    elif kind == "RESCIND_FP":
        idx = int(action["element_index"])
        conn.execute(
            "DELETE FROM corrections "
            "WHERE job_id = ? AND element_index = ? AND is_delete = 1",
            (job_id, idx),
        )
        inverse = {"action": "FP", "element_index": idx}

    elif kind == "FN_ADDED":
        bbox = action.get("bbox") or []
        if len(bbox) < 4:
            raise HTTPException(
                status_code=400,
                detail="FN_ADDED requires bbox=[x1,y1,x2,y2]",
            )
        new_entry = {
            "bbox": [float(x) for x in bbox[:4]],
            "score": 1.0,
            "source": "human_added",
        }
        cols.append(new_entry)
        new_idx = len(cols) - 1
        conn.execute(
            "INSERT OR IGNORE INTO corrections "
            "(job_id, element_type, element_index, "
            " original_element, changes, is_delete) "
            "VALUES (?, 'column', ?, '{}', ?, 0)",
            (job_id, new_idx, json.dumps({
                "bbox": new_entry["bbox"],
                "source": "human_added",
            })),
        )
        inverse = {"action": "DELETE_FN", "element_index": new_idx}

    elif kind == "DELETE_FN":
        idx = int(action["element_index"])
        if not (0 <= idx < len(cols)):
            raise HTTPException(status_code=400,
                                detail=f"element_index {idx} out of range")
        if cols[idx].get("source") != "human_added":
            raise HTTPException(
                status_code=400,
                detail="DELETE_FN target is not a human-added slot",
            )
        conn.execute(
            "DELETE FROM corrections "
            "WHERE job_id = ? AND element_index = ? AND is_delete = 0",
            (job_id, idx),
        )
        # We DO NOT truncate the JSON columns list — that would shift
        # every subsequent element_index. Instead, mark the slot as
        # "removed" via a is_delete=1 audit row tagged action=delete_fn,
        # mirroring the old app's behaviour so retrain_yolo's
        # delete-set picks it up.
        conn.execute(
            "INSERT OR REPLACE INTO corrections "
            "(job_id, element_type, element_index, "
            " original_element, changes, is_delete) "
            "VALUES (?, 'column', ?, ?, ?, 1)",
            (job_id, idx, json.dumps(cols[idx]),
             json.dumps({"action": "delete_fn"})),
        )
        inverse = {"action": "RESTORE_FN", "element_index": idx}

    elif kind == "RESTORE_FN":
        idx = int(action["element_index"])
        if not (0 <= idx < len(cols)):
            raise HTTPException(status_code=400,
                                detail=f"element_index {idx} out of range")
        conn.execute(
            "DELETE FROM corrections "
            "WHERE job_id = ? AND element_index = ? AND is_delete = 1",
            (job_id, idx),
        )
        conn.execute(
            "INSERT OR IGNORE INTO corrections "
            "(job_id, element_type, element_index, "
            " original_element, changes, is_delete) "
            "VALUES (?, 'column', ?, '{}', ?, 0)",
            (job_id, idx, json.dumps({
                "bbox": cols[idx].get("bbox"),
                "source": "human_added",
            })),
        )
        inverse = {"action": "DELETE_FN", "element_index": idx}

    else:
        raise HTTPException(status_code=400,
                            detail=f"unknown action: {kind!r}")

    # Always persist the JSON in case `cols` was mutated (FN_ADDED is
    # the only mutator today; DELETE_FN keeps the slot in place to
    # preserve element_index stability for downstream consumers).
    # det was mutated in place; meta.n keeps the canonical count.
    det["meta"] = {**det.get("meta", {}), "n": len(cols)}
    _write_px(px_path, det)
    conn.commit()

    if record_undo:
        _push_undo(job_id, action, inverse)

    return inverse, _compute_states(cols, job_id, conn)


# ──────────────────────────────────────────────────────────────────────
# HTTP routes.
# ──────────────────────────────────────────────────────────────────────


class MarkRequest(BaseModel):
    job_id: str
    action: str
    element_index: Optional[int] = None
    bbox: Optional[list[float]] = None


class JobOnlyRequest(BaseModel):
    job_id: str


class InferRequest(BaseModel):
    job_id: str
    drawing_id: str


def _db_path_from(request: Request):
    return request.app.state.config.get("db_path")


@router.get("/api/detections")
def get_detections(job_id: str, request: Request):
    """Return `{job_id, n_columns, detections: [...]}`.

    Each detection: `{element_index, bbox, state, source}`. State is
    computed via `iter_effective_corrections` so undone FP marks are
    already filtered.
    """
    px_path = JOBS_DIR / job_id / "px_detections.json"
    if not px_path.exists():
        raise HTTPException(
            status_code=412,
            detail={
                "error": "no_detections_yet",
                "job_id": job_id,
                "hint": "POST /api/infer to populate px_detections.json",
            },
        )
    with _JOB_LOCK:
        det = _read_px(px_path)
        cols = det.get("columns", [])
        conn = get_connection(_db_path_from(request))
        try:
            states = _compute_states(cols, job_id, conn)
        finally:
            conn.close()
    out = []
    for i, c in enumerate(cols):
        out.append({
            "element_index": i,
            "bbox":          c.get("bbox"),
            "score":         c.get("score", 1.0),
            "source":        c.get("source", "model"),
            "state":         states.get(i, "UNREVIEWED"),
        })
    return {"job_id": job_id, "n_columns": len(cols), "detections": out}


@router.post("/api/infer")
def post_infer(req: InferRequest, request: Request):
    """Run YOLO + post-processing on `drawing_id`, persist columns.

    Concurrent /api/infer calls for the same job are blocked behind
    `_JOB_LOCK`. Refuses (409) if the JSON already has model
    detections — re-inference is an explicit caller decision.
    """
    job_id = req.job_id
    drawing_id = req.drawing_id

    try:
        raster_path, meta = resolve_drawing(drawing_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=412, detail=str(e))

    # Weights — config override or default repo-root file.
    cfg = request.app.state.config
    weights_path: Path = (cfg.get("weights_path")
                          or cfg["project_root"] / "column_detect.pt")
    if not weights_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=(f"weights must be a regular file "
                    f"(got: {weights_path})"),
        )

    px_path = JOBS_DIR / job_id / "px_detections.json"

    with _JOB_LOCK:
        det_before = _read_px(px_path)
        cols_before = det_before.get("columns", [])
        n_model = sum(1 for c in cols_before
                      if c.get("source") != "human_added")
        if n_model > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"px_detections.json for job {job_id} already has "
                    f"{n_model} model detections. Delete the file or "
                    "relaunch with a fresh drawing-id to re-infer."
                ),
            )

    # Heavy compute OUTSIDE the lock — /api/marks can run concurrently
    # with tiled_predict; the merge below re-takes the lock for the
    # short write window only.
    from column_review.inference import run_inference
    result = run_inference(drawing_id, raster_path, weights_path, cfg)

    new_cols = [
        {"bbox": bb, "score": sc}
        for bb, sc in zip(result.boxes, result.scores)
    ]

    with _JOB_LOCK:
        det_after = _read_px(px_path)
        cols_after = det_after.get("columns", [])
        n_model_after = sum(1 for c in cols_after
                            if c.get("source") != "human_added")
        if n_model_after > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    "px_detections.json was populated by a concurrent "
                    f"/api/infer call ({n_model_after} model detections "
                    "appeared while this one was running)"
                ),
            )
        human_added = [c for c in cols_after
                       if c.get("source") == "human_added"]
        merged = human_added + new_cols
        det_after["columns"] = merged
        det_after["meta"] = {
            **det_after.get("meta", {}),
            "n":            len(merged),
            "inference_ts": time.time(),
            "device":       result.device,
            "elapsed":      result.elapsed_seconds,
        }
        _write_px(px_path, det_after)
        print(f"[infer] wrote {len(merged)} columns to {px_path}",
              flush=True)

    return {"ok": True, "n_detections": len(new_cols),
            "n_preserved": len(human_added), "job_id": job_id,
            "device": result.device,
            "elapsed_seconds": result.elapsed_seconds}


@router.post("/api/marks")
def post_marks(req: MarkRequest, request: Request):
    """Apply one mark; record the inverse on the undo stack.

    For autosave-on-action (R10) the SQL transaction commits inside
    `_apply_mark_locked` before the response returns. The whole hot
    path is timed and logged for the R11 budget verification.
    """
    px_path = JOBS_DIR / req.job_id / "px_detections.json"
    if not px_path.exists():
        raise HTTPException(status_code=404,
                            detail="px_detections.json missing")
    t0 = time.perf_counter()
    with _JOB_LOCK:
        det = _read_px(px_path)
        conn = get_connection(_db_path_from(request))
        try:
            action = {"action": req.action,
                      "element_index": req.element_index,
                      "bbox": req.bbox}
            _, states = _apply_mark_locked(
                req.job_id, action, conn, px_path, det, record_undo=True)
        finally:
            conn.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[marks] saved job={req.job_id[:8]} "
          f"action={req.action} idx={req.element_index} "
          f"in {elapsed_ms:.1f}ms", flush=True)
    return {"ok": True, "elapsed_ms": elapsed_ms,
            "n_columns": len(det.get("columns", [])),
            "states": {str(k): v for k, v in states.items()}}


@router.post("/api/undo")
def post_undo(req: JobOnlyRequest, request: Request):
    """Pop one entry from the undo stack and apply its inverse."""
    stack = _UNDO.get(req.job_id)
    if not stack:
        return {"ok": False, "reason": "undo_stack_empty"}
    original, inverse = stack.pop()
    px_path = JOBS_DIR / req.job_id / "px_detections.json"
    with _JOB_LOCK:
        det = _read_px(px_path)
        conn = get_connection(_db_path_from(request))
        try:
            _, states = _apply_mark_locked(
                req.job_id, inverse, conn, px_path, det,
                record_undo=False)
        finally:
            conn.close()
        # Record the redo entry. We push the ORIGINAL action so the
        # next redo replays exactly what the user did.
        redo_stack = _REDO.setdefault(req.job_id,
                                       deque(maxlen=_STACK_MAX))
        redo_stack.append((original, inverse))
    return {"ok": True, "applied": inverse,
            "states": {str(k): v for k, v in states.items()}}


@router.post("/api/redo")
def post_redo(req: JobOnlyRequest, request: Request):
    """Pop one entry from the redo stack and re-apply the original."""
    stack = _REDO.get(req.job_id)
    if not stack:
        return {"ok": False, "reason": "redo_stack_empty"}
    original, inverse = stack.pop()
    px_path = JOBS_DIR / req.job_id / "px_detections.json"
    with _JOB_LOCK:
        det = _read_px(px_path)
        conn = get_connection(_db_path_from(request))
        try:
            _, states = _apply_mark_locked(
                req.job_id, original, conn, px_path, det,
                record_undo=False)
        finally:
            conn.close()
        undo_stack = _UNDO.setdefault(req.job_id,
                                       deque(maxlen=_STACK_MAX))
        undo_stack.append((original, inverse))
    return {"ok": True, "applied": original,
            "states": {str(k): v for k, v in states.items()}}
