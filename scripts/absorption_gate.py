"""Post-retrain absorption gate for `column_rescue.pt`.

The gate runs after `scripts/train_yolo_rescue.py` finishes training a
fresh quarantine .pt. It verifies the new weights ACTUALLY absorbed
the latest correction batch before publishing them as
`column_rescue.pt`. If either of the two checks fails, the
quarantine path is left in place, `column_rescue.pt` is NOT
overwritten, and a structured diagnostic is written so the HITL UI
can surface it.

Two checks (spec `feedback-loop::Single-model absorption gate`):

  - **FN coverage**: for every effective `is_delete=0` correction in
    the latest batch, the new weights MUST propose at least one bbox
    with IoU ≥ `τ_fn` against the FN_ADDED bbox.

  - **FP suppression**: for every effective `is_delete=1` correction,
    the new weights MUST emit zero proposals with IoU ≥ `τ_fp`
    against the FP bbox.

"Latest batch" = every correction newer than the previous
`column_rescue.meta.json["latest_correction_ts_per_job"][job_id]`.
On a never-trained system, the batch is every correction in the DB.

Public API:
    run_gate(quarantine_path, project_root, *, tau_fn, tau_fp) -> dict
        Returns the meta-json-shaped result dict. Does NOT raise on
        gate failure (caller decides). The result includes
        `gate_status` (`"passed"` / `"failed"`),
        `latest_correction_ts_per_job` (per-job ts map),
        and on failure a `gate_failure` block listing every offending
        correction.

CLI:
    python3 scripts/absorption_gate.py <quarantine.pt>
        Loads the quarantine weights, runs the gate, prints the
        decision, exits 0 on pass / 2 on fail.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
# Idempotent path inserts so re-imports across long-lived processes
# (e.g., the column-review server) don't grow sys.path on every
# call. Module top is the only insert point — the previous in-function
# `sys.path.insert` inside `_predict_tiles_batched` leaked one
# duplicate entry per gate-job invocation.
for _p in (_SCRIPTS_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tile_geometry import (   # noqa: E402
    TILE_SIZE, bbox_at_index, iou_xyxy, tile_origin_for_bbox,
)
from column_review.yolo_rescue import (   # noqa: E402
    get_rescue_batch_chunk, unpack_yolo_result,
)

DATA_ROOT = _PROJECT_ROOT / "data"
JOBS_DIR  = DATA_ROOT / "jobs"
CORR_DB   = DATA_ROOT / "corrections.db"

# Defaults — overridable via CLI / config.yaml.
TAU_FN_DEFAULT = 0.5
TAU_FP_DEFAULT = 0.3


def _read_previous_meta(project_root: Path) -> dict:
    meta_path = project_root / "column_rescue.meta.json"
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _latest_batch(prev_per_job: dict) -> list[dict]:
    """Every correction in the DB whose timestamp exceeds the per-job
    threshold from the previous training run. Returns enriched dicts
    with bbox + tile origin + kind ready for the inference probe.
    """
    if not CORR_DB.exists():
        return []
    from corrections_logger import iter_effective_corrections   # noqa: E402

    out: list[dict] = []
    conn = sqlite3.connect(str(CORR_DB))
    try:
        for r in iter_effective_corrections(conn):
            job_id, _et, idx, _orig, _changes, is_delete, ts = r
            threshold = float(prev_per_job.get(job_id, 0) or 0)
            if float(ts) <= threshold:
                continue
            out.append({
                "job_id":        job_id,
                "element_index": int(idx),
                "is_delete":     int(is_delete),
                "timestamp":     float(ts),
            })
        # tp_confirmations rows count as FN-coverage positives. Same
        # threshold logic.
        for job_id, idx, ts in conn.execute(
            "SELECT job_id, element_index, ts FROM tp_confirmations"
        ).fetchall():
            threshold = float(prev_per_job.get(job_id, 0) or 0)
            if float(ts) <= threshold:
                continue
            out.append({
                "job_id":        job_id,
                "element_index": int(idx),
                "is_delete":     0,
                "timestamp":     float(ts),
            })
    finally:
        conn.close()
    return out


def _load_job_cols(job_id: str, cache: dict[str, list | None]
                   ) -> list | None:
    """Return `cols` for a job, cached per-process. Returns None and
    caches None when the job's px_detections.json is missing or
    unparseable. One json.loads per job, not per correction."""
    if job_id in cache:
        return cache[job_id]
    pp = JOBS_DIR / job_id / "px_detections.json"
    if not pp.is_file():
        cache[job_id] = None
        return None
    try:
        cols = json.loads(pp.read_text()).get("columns", [])
    except (OSError, json.JSONDecodeError):
        cols = None
    cache[job_id] = cols
    return cols


def _predict_tiles_batched(model, img: Image.Image,
                           tile_origins: list[tuple[int, int]],
                           conf_threshold: float
                           ) -> list[list[tuple[float, float, float, float, float]]]:
    """Stream-chunk crop + predict + origin-shift over `tile_origins`.

    Outer loop chunks `tile_origins` at `get_rescue_batch_chunk()` so
    only `chunk_size` PIL crops are materialised at a time (peak host
    RAM bounded). Each chunk drives ONE `model.predict` call directly;
    `unpack_yolo_result` converts each Result to tile-local
    `(x1, y1, x2, y2, score)` tuples, which are then shifted into
    drawing-pixel coords by the originating tile's (x0, y0).

    The chunking POLICY (env var + default) is shared with the
    standalone `predict_chunked` helper via `get_rescue_batch_chunk`;
    using `model.predict` here rather than re-entering `predict_chunked`
    avoids a degenerate inner chunk loop (we already chunk by design).
    """
    chunk_size = get_rescue_batch_chunk()
    out: list[list[tuple[float, float, float, float, float]]] = []
    for i in range(0, len(tile_origins), chunk_size):
        chunk_origins = tile_origins[i:i + chunk_size]
        tiles = [img.crop((x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE))
                 for x0, y0 in chunk_origins]
        results = model.predict(tiles, conf=conf_threshold,
                                verbose=False)
        for (x0, y0), r in zip(chunk_origins, results):
            preds = unpack_yolo_result(r)
            out.append([(x1 + x0, y1 + y0, x2 + x0, y2 + y0, c)
                        for (x1, y1, x2, y2, c) in preds])
    return out


def run_gate(quarantine_path: Path, project_root: Path, *,
             tau_fn: float = TAU_FN_DEFAULT,
             tau_fp: float = TAU_FP_DEFAULT,
             conf_threshold: float = 0.25,
             ) -> dict:
    """Run both gate checks against `quarantine_path`. Returns the
    meta-shaped result dict. Does not move files — the caller decides
    what to do based on `gate_status`.

    Lazy-imports ultralytics so the module import stays cheap.
    """
    prev_meta = _read_previous_meta(project_root)
    prev_per_job = prev_meta.get("latest_correction_ts_per_job") or {}

    batch = _latest_batch(prev_per_job)
    fn_targets = [c for c in batch if not c["is_delete"]]
    fp_targets = [c for c in batch if c["is_delete"]]

    # Resolve bbox per correction with a per-job cache so we json.loads
    # each px_detections.json exactly once even when a job contributes
    # many corrections.
    cols_cache: dict[str, list | None] = {}
    by_job: dict[str, list[dict]] = {}
    for c in batch:
        cols = _load_job_cols(c["job_id"], cols_cache)
        c["bbox"] = (bbox_at_index(cols, c["element_index"])
                     if cols is not None else None)
        if c["bbox"] is not None:
            by_job.setdefault(c["job_id"], []).append(c)

    if not batch:
        # First run on empty corrections — vacuously passed.
        return {
            "gate_status":                   "passed",
            "latest_correction_ts_per_job":  dict(prev_per_job),
            "n_fn_targets":                  0,
            "n_fp_targets":                  0,
            "tau_fn":                        tau_fn,
            "tau_fp":                        tau_fp,
            "checked_ts":                    time.time(),
        }

    from ultralytics import YOLO
    model = YOLO(str(quarantine_path))

    fn_failures: list[dict] = []
    fp_failures: list[dict] = []

    for job_id, targets in by_job.items():
        render_path = JOBS_DIR / job_id / "render.jpg"
        if not render_path.is_file():
            # No render to probe — gate cannot prove absorption.
            for c in targets:
                rec = {"job_id": job_id,
                       "element_index": c["element_index"],
                       "bbox": c["bbox"], "best_iou": None,
                       "reason": "render.jpg missing"}
                (fp_failures if c["is_delete"] else fn_failures).append(rec)
            continue
        with Image.open(render_path) as img:
            W, H = img.size
            img.load()
            origins = [tile_origin_for_bbox(c["bbox"], W, H)
                       for c in targets]
            tile_preds = _predict_tiles_batched(
                model, img, origins, conf_threshold)
            for c, preds in zip(targets, tile_preds):
                best_iou = max(
                    (iou_xyxy(c["bbox"], (p[0], p[1], p[2], p[3]))
                     for p in preds),
                    default=0.0,
                )
                if c["is_delete"]:
                    # FP — must have ZERO predictions at IoU ≥ tau_fp.
                    if best_iou >= tau_fp:
                        fp_failures.append({
                            "job_id":        job_id,
                            "element_index": c["element_index"],
                            "bbox":          c["bbox"],
                            "best_iou":      best_iou,
                        })
                else:
                    # FN — must have AT LEAST ONE prediction at IoU ≥ tau_fn.
                    if best_iou < tau_fn:
                        fn_failures.append({
                            "job_id":        job_id,
                            "element_index": c["element_index"],
                            "bbox":          c["bbox"],
                            "best_iou":      best_iou,
                        })

    # Compute the latest-correction-ts map regardless of pass/fail —
    # the caller writes it into meta.json only on pass.
    latest_per_job = dict(prev_per_job)
    for c in batch:
        cur = latest_per_job.get(c["job_id"], 0)
        if c["timestamp"] > cur:
            latest_per_job[c["job_id"]] = c["timestamp"]

    status = "passed" if not fn_failures and not fp_failures else "failed"
    result = {
        "gate_status":                  status,
        "latest_correction_ts_per_job": latest_per_job,
        "n_fn_targets":                 len(fn_targets),
        "n_fp_targets":                 len(fp_targets),
        "tau_fn":                       tau_fn,
        "tau_fp":                       tau_fp,
        "checked_ts":                   time.time(),
    }
    if status == "failed":
        result["gate_failure"] = {
            "fn_failures": fn_failures,
            "fp_failures": fp_failures,
            "summary": (
                f"{len(fn_failures)} FN(s) not absorbed at IoU≥{tau_fn}; "
                f"{len(fp_failures)} FP(s) still firing at IoU≥{tau_fp}"
            ),
        }
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("quarantine", help="Path to the candidate "
                                       "column_rescue_quarantine_*.pt")
    p.add_argument("--tau-fn", type=float, default=TAU_FN_DEFAULT)
    p.add_argument("--tau-fp", type=float, default=TAU_FP_DEFAULT)
    p.add_argument("--conf",   type=float, default=0.25,
                   help="conf threshold for the rescue probe")
    args = p.parse_args()
    q = Path(args.quarantine).resolve()
    if not q.is_file():
        print(f"ERROR: quarantine not found at {q}", file=sys.stderr)
        return 2
    result = run_gate(q, _PROJECT_ROOT,
                      tau_fn=args.tau_fn, tau_fp=args.tau_fp,
                      conf_threshold=args.conf)
    print(json.dumps(result, indent=2))
    return 0 if result["gate_status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
