"""Detections, marks, undo/redo, and inference routes.

Routes:
    GET  /api/detections?job_id=X → per-detection list with state
    POST /api/infer               → run YOLO on a drawing's raster,
                                      write px_detections.json
    POST /api/marks               → apply ONE mark
                                      ({FP, RESCIND_FP, FN_ADDED,
                                        DELETE_FN, RESTORE_FN})
    POST /api/undo                → pop the latest mark for this job,
                                      apply its inverse, push to redo
    POST /api/redo                → pop from redo, re-apply, push to
                                      undo

The marks encoding mirrors the on-disk shape that
`scripts/retrain_yolo.py` already consumes:

  FP   → `corrections` row with `is_delete=1`, positive
         `element_index` matching the model detection. On a
         human_added slot, the existing `is_delete=0` row is
         DELETEd first so `iter_effective_corrections`'
         rescind-on-read invariant does not silently drop the FP.
  FN   → APPEND `{"bbox": [...], "source": "human_added"}` to
         `data/jobs/<job_id>/px_detections.json["columns"]`,
         then write a `corrections` row with `is_delete=0` at
         the newly-assigned positive `element_index`. The
         assigned index is stashed back into the action dict so
         a subsequent redo (after undo → DELETE_FN) replays
         through RESTORE_FN rather than re-appending a duplicate
         slot at len(cols).
  DELETE_FN → `is_delete=1` row tagged `changes={"action":
              "delete_fn"}`. `_compute_states` distinguishes
              this from a plain FP and reports state="REMOVED";
              the frontend hides REMOVED slots.

Concurrency: a single `threading.RLock` per process serialises the
read-modify-write cycle across `/api/infer`, `/api/marks`, `/api/undo`,
`/api/redo`, and `GET /api/detections` so a reader never observes a
torn intermediate state.

Session enforcement: `/api/marks`, `/api/undo`, `/api/redo` require
a `session_id` that exists in `reviewer_sessions`. Stale tabs or
curl scripts cannot write orphan corrections rows.
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


# Re-entrant per-process write lock. RLock because `_apply_mark_locked`
# calls `_compute_states` while already holding the lock; a plain Lock
# would deadlock on the inner acquire.
_JOB_LOCK = threading.RLock()

# Per-job undo/redo stacks, capped at 100 entries per the R9 spec.
# Each stack entry is `(original_action_dict, inverse_action_dict)`.
_UNDO: dict[str, deque] = {}
_REDO: dict[str, deque] = {}
_STACK_MAX = 100


# ──────────────────────────────────────────────────────────────────────
# Mark application primitives.
# ──────────────────────────────────────────────────────────────────────


def _assert_columns_well_formed(cols, px_path: Path) -> None:
    """Loud-fail on non-dict entries in `columns`.

    The schema requires each entry of `columns` be a JSON object. A
    malformed `px_detections.json` (partial migration, hand edit, or
    third-party tool) can carry `null` / strings / arrays instead, and
    `c.get("source")` on those raises AttributeError with an opaque
    detail. Raising here lets callers assume every element is a dict —
    silently filtering would shift element_index for every
    corrections-table row past the dropped slot.
    """
    bad = [i for i, c in enumerate(cols) if not isinstance(c, dict)]
    if not bad:
        return
    n = len(bad)
    sample = bad[:5]
    raise HTTPException(
        status_code=500,
        detail=(
            f"px_detections.json at {px_path} has {n} malformed "
            f"(non-dict) column entr{'y' if n == 1 else 'ies'} at "
            f"element_index {sample}"
            f"{'' if n <= 5 else f' (and {n - 5} more)'}. "
            "Inspect the file manually; the corrections-table rows "
            "reference element_index positionally, so silently "
            "compacting the list would re-target every row past a "
            "dropped slot."
        ),
    )


def _read_px(px_path: Path) -> dict:
    if not px_path.exists():
        return {"columns": [], "meta": {}}
    try:
        data = json.loads(px_path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"px_detections.json corrupt at {px_path}: {e}",
        )
    _assert_columns_well_formed(data.get("columns", []), px_path)
    return data


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
      "REMOVED"   — `is_delete=1` row tagged `changes.action=="delete_fn"`
                    (user undid their own FN add)
      "FP"        — any other `is_delete=1` row
      "FN_ADDED"  — model never produced this; cols entry has
                    `source="human_added"`
      "UNREVIEWED" — default for model detections without corrections

    `iter_effective_corrections` rescinds is_delete=1 rows that have a
    same-key is_delete=0 row. The FP mark path strips that is_delete=0
    row first when targeting a human_added slot so the FP row survives.
    """
    state: dict[int, str] = {}
    for i, c in enumerate(cols):
        state[i] = ("FN_ADDED" if c.get("source") == "human_added"
                    else "UNREVIEWED")
    for row in iter_effective_corrections(conn, job_id=job_id):
        _job, _et, idx, _orig, changes_json, is_delete, _ts = row
        if not is_delete:
            continue
        try:
            changes = json.loads(changes_json) if changes_json else {}
        except (ValueError, TypeError):
            changes = {}
        if changes.get("action") == "delete_fn":
            state[idx] = "REMOVED"
        else:
            state[idx] = "FP"
    return state


def _removed_human_indices(conn: sqlite3.Connection, job_id: str,
                           cols: list) -> set[int]:
    """Indices of human_added slots whose entry has an is_delete=1 row.

    Used by FN_ADDED dedup so re-adding at the same rounded centre as
    a previously-deleted slot is NOT silently no-op'd against the
    stale JSON entry.
    """
    out: set[int] = set()
    for (rmidx,) in conn.execute(
            "SELECT element_index FROM corrections "
            "WHERE job_id = ? AND is_delete = 1",
            (job_id,)).fetchall():
        if (0 <= rmidx < len(cols)
                and cols[rmidx].get("source") == "human_added"):
            out.add(rmidx)
    return out


def _push_undo(job_id: str, original: dict,
               inverse: Optional[dict]) -> None:
    """Record one applied mark for later undo.

    Bounded LIFO — oldest entries drop off the bottom past the
    spec's 100-level guarantee. `inverse=None` is a deliberate no-op
    (e.g., FN_ADDED dedup): nothing actually changed, so no undo
    entry is needed.
    """
    if inverse is None:
        return
    stack = _UNDO.setdefault(job_id, deque(maxlen=_STACK_MAX))
    stack.append((original, inverse))
    # A new mark invalidates any pending redo stack — the user has
    # branched away from the previously-undone history.
    _REDO.pop(job_id, None)


def _apply_mark_locked(job_id: str, action: dict,
                       conn: sqlite3.Connection,
                       px_path: Path, det: dict,
                       record_undo: bool) -> tuple[Optional[dict],
                                                    dict[int, str],
                                                    bool]:
    """Apply ONE action to the DB / JSON.

    Returns `(inverse, states, mutated_cols)`.
      inverse — the action that undoes this one, or None for a no-op
                mark (deduped FN_ADDED). The caller pushes the
                (action, inverse) pair onto the undo stack when
                `record_undo=True`.
      states  — fresh per-index state map for the response.
      mutated_cols — True iff `cols` was mutated (FN_ADDED fresh
                     append). The caller skips the JSON rewrite when
                     False — this is the dominant cost reduction for
                     the 100 ms-per-mark latency floor.

    `det` is the parsed `px_detections.json` dict; `det["columns"]` is
    mutated in place by fresh FN_ADDED only.
    """
    cols = det.setdefault("columns", [])
    kind = action.get("action")
    inverse: Optional[dict] = None
    mutated_cols = False

    if kind == "FP":
        idx = int(action["element_index"])
        if not (0 <= idx < len(cols)):
            raise HTTPException(status_code=400,
                                detail=f"element_index {idx} out of range")
        # If the slot is human_added, its is_delete=0 row would
        # rescind any FP we insert (iter_effective_corrections drops
        # is_delete=1 rows that have a same-key is_delete=0). Strip
        # the rescind first so the FP audit row survives.
        if cols[idx].get("source") == "human_added":
            conn.execute(
                "DELETE FROM corrections "
                "WHERE job_id = ? AND element_index = ? AND is_delete = 0",
                (job_id, idx),
            )
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
        new_bbox = [float(x) for x in bbox[:4]]

        # Redo path: a prior apply stashed `element_index` back into
        # this action dict before it was pushed onto the undo stack.
        # If the slot is still in cols as a human_added entry (the
        # DELETE_FN inverse keeps the slot to preserve element_index
        # stability), restore it instead of appending a duplicate at
        # len(cols).
        explicit_idx = action.get("element_index")
        if explicit_idx is not None and isinstance(explicit_idx, int):
            if (0 <= explicit_idx < len(cols)
                    and cols[explicit_idx].get("source") == "human_added"):
                conn.execute(
                    "DELETE FROM corrections "
                    "WHERE job_id = ? AND element_index = ? AND is_delete = 1",
                    (job_id, explicit_idx),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO corrections "
                    "(job_id, element_type, element_index, "
                    " original_element, changes, is_delete) "
                    "VALUES (?, 'column', ?, '{}', ?, 0)",
                    (job_id, explicit_idx,
                     json.dumps({"bbox": cols[explicit_idx].get("bbox"),
                                 "source": "human_added"})),
                )
                inverse = {"action": "DELETE_FN",
                           "element_index": explicit_idx}
            else:
                # Slot was wiped from cols somehow; fall through.
                explicit_idx = None

        if explicit_idx is None:
            # Fresh FN_ADDED — dedup at same rounded image-pixel centre.
            cx = (new_bbox[0] + new_bbox[2]) / 2.0
            cy = (new_bbox[1] + new_bbox[3]) / 2.0
            rcx, rcy = round(cx), round(cy)
            removed_human = _removed_human_indices(conn, job_id, cols)
            deduped_idx: Optional[int] = None
            for ei, existing in enumerate(cols):
                if existing.get("source") != "human_added":
                    continue
                if ei in removed_human:
                    continue
                eb = existing.get("bbox") or []
                if len(eb) < 4:
                    continue
                if (round((eb[0] + eb[2]) / 2.0) == rcx
                        and round((eb[1] + eb[3]) / 2.0) == rcy):
                    deduped_idx = ei
                    break
            if deduped_idx is not None:
                # No-op: live human_added entry already covers this
                # centre. inverse stays None → no undo entry pushed.
                pass
            else:
                new_entry = {"bbox": new_bbox, "score": 1.0,
                             "source": "human_added"}
                cols.append(new_entry)
                new_idx = len(cols) - 1
                mutated_cols = True
                # Stash the assigned index back into the action dict so
                # a later redo through this same dict routes through
                # the restore path above, not a duplicate-append.
                action["element_index"] = new_idx
                conn.execute(
                    "INSERT OR IGNORE INTO corrections "
                    "(job_id, element_type, element_index, "
                    " original_element, changes, is_delete) "
                    "VALUES (?, 'column', ?, '{}', ?, 0)",
                    (job_id, new_idx,
                     json.dumps({"bbox": new_bbox,
                                 "source": "human_added"})),
                )
                inverse = {"action": "DELETE_FN",
                           "element_index": new_idx}

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
        # Drop the is_delete=0 row first so the new is_delete=1 row
        # is not rescinded by it.
        conn.execute(
            "DELETE FROM corrections "
            "WHERE job_id = ? AND element_index = ? AND is_delete = 0",
            (job_id, idx),
        )
        # Tag the audit row so `_compute_states` distinguishes
        # "user undid their own add" (REMOVED) from "user marked the
        # slot as FP" (visible).
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

    if mutated_cols:
        det["meta"] = {**det.get("meta", {}), "n": len(cols)}
        _write_px(px_path, det)
    conn.commit()

    if record_undo:
        _push_undo(job_id, action, inverse)

    return inverse, _compute_states(cols, job_id, conn), mutated_cols


# ──────────────────────────────────────────────────────────────────────
# HTTP routes.
# ──────────────────────────────────────────────────────────────────────


class MarkRequest(BaseModel):
    job_id: str
    session_id: str
    action: str
    element_index: Optional[int] = None
    bbox: Optional[list[float]] = None


class JobSessionRequest(BaseModel):
    job_id: str
    session_id: str


class InferRequest(BaseModel):
    job_id: str
    drawing_id: str


def _db_path_from(request: Request):
    return request.app.state.config.get("db_path")


def _require_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Raise 409 if `session_id` is not in `reviewer_sessions`.

    Closes the gap where a stale browser tab, a curl script, or any
    other caller could POST `/api/marks` with a guessed `job_id` and
    write corrections rows without a reviewer-id provenance row.
    """
    if not session_id:
        raise HTTPException(
            status_code=409,
            detail="session_id is required; POST /api/open first",
        )
    row = conn.execute(
        "SELECT 1 FROM reviewer_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=409,
            detail=(
                f"session_id {session_id[:8]}… is not registered. "
                "POST /api/open to start a fresh session."
            ),
        )


def _detection_view(idx: int, c: dict, states: dict[int, str]) -> dict:
    """Serialise one detection for the wire."""
    return {
        "element_index": idx,
        "bbox":          c.get("bbox"),
        "score":         c.get("score", 1.0),
        "source":        c.get("source", "model"),
        "state":         states.get(idx, "UNREVIEWED"),
    }


@router.get("/api/detections")
def get_detections(job_id: str, request: Request):
    """Return `{job_id, n_columns, detections: [...]}`.

    Each detection: `{element_index, bbox, score, source, state}`.
    State is computed via `iter_effective_corrections` so undone FP
    marks are already filtered.
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
    out = [_detection_view(i, c, states) for i, c in enumerate(cols)]
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

    Returns `{ok, elapsed_ms, n_columns, states, new_detection}`.
    `new_detection` is non-null only when FN_ADDED grew the columns
    list — the frontend pushes it to local state and avoids a full
    `/api/detections` refetch (R10 efficiency, R11 budget).
    """
    px_path = JOBS_DIR / req.job_id / "px_detections.json"
    if not px_path.exists():
        raise HTTPException(status_code=404,
                            detail="px_detections.json missing")
    t0 = time.perf_counter()
    new_detection_idx: Optional[int] = None
    with _JOB_LOCK:
        det = _read_px(px_path)
        n_before = len(det.get("columns", []))
        conn = get_connection(_db_path_from(request))
        try:
            _require_session(conn, req.session_id)
            action = {"action": req.action,
                      "element_index": req.element_index,
                      "bbox": req.bbox}
            _, states, mutated = _apply_mark_locked(
                req.job_id, action, conn, px_path, det,
                record_undo=True)
        finally:
            conn.close()
        n_after = len(det.get("columns", []))
        if mutated and n_after > n_before:
            new_detection_idx = n_after - 1
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[marks] saved job={req.job_id[:8]} "
          f"action={req.action} idx={req.element_index} "
          f"in {elapsed_ms:.1f}ms", flush=True)

    new_detection = None
    if new_detection_idx is not None:
        c = det["columns"][new_detection_idx]
        new_detection = _detection_view(new_detection_idx, c, states)

    return {"ok": True, "elapsed_ms": elapsed_ms,
            "n_columns": len(det.get("columns", [])),
            "new_detection": new_detection,
            "states": {str(k): v for k, v in states.items()}}


@router.post("/api/undo")
def post_undo(req: JobSessionRequest, request: Request):
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
            _require_session(conn, req.session_id)
            _, states, _mutated = _apply_mark_locked(
                req.job_id, inverse, conn, px_path, det,
                record_undo=False)
        finally:
            conn.close()
        redo_stack = _REDO.setdefault(req.job_id,
                                       deque(maxlen=_STACK_MAX))
        redo_stack.append((original, inverse))
    return {"ok": True, "applied": inverse,
            "n_columns": len(det.get("columns", [])),
            "states": {str(k): v for k, v in states.items()}}


@router.post("/api/redo")
def post_redo(req: JobSessionRequest, request: Request):
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
            _require_session(conn, req.session_id)
            _, states, _mutated = _apply_mark_locked(
                req.job_id, original, conn, px_path, det,
                record_undo=False)
        finally:
            conn.close()
        undo_stack = _UNDO.setdefault(req.job_id,
                                       deque(maxlen=_STACK_MAX))
        undo_stack.append((original, inverse))
    return {"ok": True, "applied": original,
            "n_columns": len(det.get("columns", [])),
            "states": {str(k): v for k, v in states.items()}}
