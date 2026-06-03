#!/usr/bin/env python3
"""
YOLO Fine-tuning Script — Training Flywheel (column detector edition)

Adapted from the mcc-amplify-v5 5-class flywheel for THIS project's single-
class column detector. Each user correction captured by a CorrectionsLogger
becomes one labelled training example:

  - Deleted columns   → false-positive bbox (excluded from positive labels)
  - Corrected columns → confirmed bbox label

Prerequisites — NOT yet present in 1-column-train. Set these up before
running this script:

    data/corrections.db                     — corrections SQLite log
    data/jobs/{job_id}/render.jpg           — rendered floor plan image
    data/jobs/{job_id}/px_detections.json   — pixel-space YOLO detections

Usage (run from project root):
    python scripts/retrain_yolo.py
    python scripts/retrain_yolo.py --epochs 30 --min-corrections 20

Output:
    data/yolo_finetune/                          — YOLO dataset structure
    runs/detect/correction_feedback/weights/best.pt
    column_detect_ft_{timestamp}.pt              — copy of best.pt at project root
"""

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from _yolo_dataset_utils import init_yolo_dataset_dirs, write_data_yaml


METRICS_DIR = Path("data/metrics")


def _sha1_of_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _sha1_of_dir(path: Path) -> str | None:
    """Stable directory digest: SHA1 of sorted (relpath, file SHA1) pairs."""
    if not path.exists() or not path.is_dir():
        return None
    pairs = []
    for p in sorted(path.rglob("*")):
        if p.is_file():
            pairs.append(f"{p.relative_to(path)}::{_sha1_of_file(p)}")
    h = hashlib.sha1("\n".join(pairs).encode()).hexdigest()
    return h[:12]


def _evaluate_regression_tgch(model, project_root: Path) -> dict:
    """Run TILED inference + the deployed post-processing pipeline on
    TGCH-TD-S-200-L3-00 and report against the expected 440 column
    instances.

    The audit metric must mirror deployed inference, so we run BOTH the
    tiled predict AND the 7-filter post-processing. Recording raw vs
    filtered separately surfaces regressions where the head changes
    behaviour but the filters compensate (or vice versa).

    Returns the populated regression dict with `expected`, `raw_detected`,
    `detected` (post-processed), `recall`, etc.
    """
    import numpy as np
    from PIL import Image
    from tiled_inference import tiled_predict
    from postprocess_pipeline import run_pipeline, DEFAULT_CONFIG

    candidates = [
        Path("/home/jiezhi/Documents/TGCH floor plan/L3.jpg"),
        project_root / "data" / "raw" / "drawings" / "TGCH-TD-S-200-L3-00.png",
    ]
    plan = next((p for p in candidates if p.exists()), None)
    if plan is None:
        return {"expected": 440, "detected": 0, "recall": 0.0,
                "note": "TGCH-TD-S-200-L3-00 plan not found at known paths"}

    _prev_max = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(plan).convert("RGB")
        boxes, scores, tile_counts = tiled_predict(
            model, img, tile=1280, step=1080, conf=0.25, iou=0.45,
        )
        raw_detected = len(boxes)
        img_gray = np.asarray(img.convert("L"))
        boxes_final, _scores_final, _audit = run_pipeline(
            img_gray, boxes, scores,
            config=DEFAULT_CONFIG,
            input_dpi=300,
            tile_detection_counts=tile_counts,
        )
        detected = len(boxes_final)
    except Exception as e:
        return {"expected": 440, "detected": 0, "recall": 0.0,
                "note": f"tiled inference / post-processing failed: {e}"}
    finally:
        Image.MAX_IMAGE_PIXELS = _prev_max
    return {
        "expected":     440,
        "raw_detected": int(raw_detected),
        "detected":     int(detected),
        "recall":       float(detected) / 440.0,
        "raw_recall":   float(raw_detected) / 440.0,
        "n_tiles":      len(tile_counts),
        "source":       str(plan),
    }


def _write_metrics(model, train_results, args, n_corrections, project_root: Path,
                   val_split_strategy: str = "unknown"):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    metrics_box = getattr(train_results, "results_dict", {}) if train_results else {}
    map50    = float(metrics_box.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(metrics_box.get("metrics/mAP50-95(B)", 0.0))
    precision = float(metrics_box.get("metrics/precision(B)", 0.0))
    recall    = float(metrics_box.get("metrics/recall(B)", 0.0))

    pool_dir = project_root / "data" / "hard_negatives"
    pool_manifest = pool_dir / "manifest.json"
    n_hard_negatives = 0
    if pool_manifest.exists():
        try:
            n_hard_negatives = int(json.loads(pool_manifest.read_text()).get("n_entries", 0))
        except Exception:
            pass

    metrics = {
        "revision":                str(ts),
        "ts":                      ts,
        "epochs":                  args.epochs,
        "base_weights":            str(args.base_weights),
        "base_weights_sha":        _sha1_of_file(Path(args.base_weights)),
        "jobs_dir_sha":            _sha1_of_dir(Path("data/jobs")),
        "hard_negatives_dir_sha":  _sha1_of_dir(pool_dir),
        "n_corrections_consumed":  n_corrections,
        "n_hard_negatives":        n_hard_negatives,
        "val_split_strategy":      val_split_strategy,
        "fp_rate_per_drawing":     None,   # populated downstream by analysis
        "regression": {
            "tgch_td_s_200_l3_00": _evaluate_regression_tgch(model, project_root),
        },
    }

    # mAP fields land under their normative key ONLY when val is genuinely
    # held out. When train == val (duplicated_train), report under an
    # explicit `_train_leaked` suffix so dashboards comparing mAP50 across
    # revisions cannot accidentally mix the two distributions.
    if val_split_strategy == "duplicated_train":
        metrics["mAP50_train_leaked"]    = map50
        metrics["mAP50_95_train_leaked"] = map50_95
        metrics["precision_train_leaked"] = precision
        metrics["recall_train_leaked"]    = recall
        metrics["mAP50"]    = None
        metrics["mAP50_95"] = None
        metrics["precision"] = None
        metrics["recall"]    = None
    else:
        metrics["mAP50"]    = map50
        metrics["mAP50_95"] = map50_95
        metrics["precision"] = precision
        metrics["recall"]    = recall

    out_path = METRICS_DIR / f"{ts}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics written: {out_path}")
    if val_split_strategy == "duplicated_train":
        print("  mAP fields written under '*_train_leaked' suffix "
              "(val was duplicated from train).")
    else:
        print(f"  mAP50={map50:.3f}  mAP50_95={map50_95:.3f}  "
              f"P={precision:.3f}  R={recall:.3f}")
    reg = metrics["regression"]["tgch_td_s_200_l3_00"]
    print(f"  regression TGCH-TD-S-200-L3-00: raw={reg.get('raw_detected', 0)} / "
          f"filtered={reg['detected']} / expected={reg['expected']} "
          f"(recall {reg['recall']:.3f})")
    return out_path


# ── Single-class scheme (matches dataset/column/data.yaml) ───────────────────
CLASS_NAMES = ["column"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Accept both plural ("columns" from recipe) and singular ("column" from logger).
TYPE_TO_CLASS = {"columns": "column", "column": "column"}


def load_corrections(db_path: Path) -> list:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT job_id, element_type, element_index, "
        "       original_element, changes, is_delete "
        "FROM corrections ORDER BY timestamp"
    ).fetchall()
    conn.close()
    return [
        {
            "job_id":        r[0],
            "element_type":  r[1],
            "element_index": r[2],
            "original":      json.loads(r[3]),
            "changes":       json.loads(r[4]),
            "is_delete":     bool(r[5]),
        }
        for r in rows
    ]


def _copy_golden_set(dataset_dir: Path) -> int:
    """If `data/golden/{images,labels}/` exists, copy it into the val
    split. The golden set is the audit hold-out — corrections never
    enter it. Returns the number of golden images copied.
    """
    golden_root = Path("data/golden")
    g_img = golden_root / "images"
    g_lbl = golden_root / "labels"
    if not g_img.exists() or not g_lbl.exists():
        return 0
    n = 0
    for img_path in sorted(g_img.iterdir()):
        if not img_path.is_file():
            continue
        stem = img_path.stem
        lbl_path = g_lbl / f"{stem}.txt"
        if not lbl_path.exists():
            continue
        shutil.copy2(img_path, dataset_dir / "images" / "val" / img_path.name)
        shutil.copy2(lbl_path, dataset_dir / "labels" / "val" / lbl_path.name)
        n += 1
    return n


def build_dataset(corrections: list, dataset_dir: Path,
                  val_fraction: float = 0.2) -> tuple[int, str]:
    """
    Build YOLO-format dataset from corrections + per-job checkpoint data.

    Returns (n_unique_images, val_split_strategy) where strategy is one of:
        - "golden"           — data/golden/ used as val; corrections in train.
        - "job_split"        — val carved from the corrections jobs.
        - "duplicated_train" — single job duplicated into both splits;
                                mAP from this run is in-distribution and
                                MUST be reported under a different JSON key.

    Val-split strategy (in priority order):
      1. If `data/golden/{images,labels}/` exists, use it as the val set;
         all corrections go to train. This is the auditable held-out
         path — corrections never enter golden.
      2. Otherwise, if multiple jobs exist, partition `val_fraction` of
         jobs into val and the rest into train.
      3. Otherwise (single-job, no golden), duplicate the only job into
         BOTH train and val with a clear warning. mAP will be optimistic
         (in-distribution val) but ultralytics needs SOME val set.
    """
    from PIL import Image as _PIL

    init_yolo_dataset_dirs(dataset_dir)

    n_golden = _copy_golden_set(dataset_dir)
    use_golden = n_golden > 0

    jobs: dict[str, list] = {}
    for c in corrections:
        jobs.setdefault(c["job_id"], []).append(c)

    job_ids = sorted(jobs.keys())

    duplicate_into_both = False
    if use_golden:
        val_set = set()  # every job goes to train; val is the golden set
        strategy = "golden"
        print(f"  Using data/golden/ as val set ({n_golden} images held out).")
    elif len(job_ids) <= 1:
        # No golden, only one job — write the job into BOTH train and val
        # so ultralytics has a non-empty val to compute mAP. mAP will be
        # optimistic; the warning makes that explicit AND _write_metrics
        # records val_split_strategy='duplicated_train' so consumers can
        # filter the deployed-mAP fields out.
        val_set = set()  # not via val_set; explicit duplication below
        duplicate_into_both = True
        strategy = "duplicated_train"
        print("  WARNING: only one job_id and no data/golden/. Duplicating "
              "the job into both train and val. mAP will be optimistic. "
              "Create data/golden/ for a real held-out evaluation.")
    else:
        n_val = max(1, min(int(len(job_ids) * val_fraction), len(job_ids) - 1))
        val_set = set(job_ids[:n_val])
        strategy = "job_split"

    valid_count = n_golden   # golden images already count as added

    for job_id, job_corrections in jobs.items():
        splits_for_job = (
            ("train", "val") if duplicate_into_both
            else (("val",) if job_id in val_set else ("train",))
        )
        render_path = Path(f"data/jobs/{job_id}/render.jpg")
        detect_path = Path(f"data/jobs/{job_id}/px_detections.json")

        if not render_path.exists():
            print(f"  [skip] {job_id[:8]}… — render.jpg not found")
            continue
        if not detect_path.exists():
            print(f"  [skip] {job_id[:8]}… — px_detections.json not found")
            continue

        with open(detect_path) as f:
            detections = json.load(f)

        deleted = {
            (c["element_type"], c["element_index"])
            for c in job_corrections if c["is_delete"]
        }

        # Copy render into each target split (one split normally; two when
        # duplicate_into_both is True).
        for split in splits_for_job:
            dest_img = dataset_dir / "images" / split / f"{job_id}.jpg"
            shutil.copy2(render_path, dest_img)

        with _PIL.open(render_path) as img:
            img_w, img_h = img.size

        labels = []

        for el_type, elements in detections.items():
            class_name = TYPE_TO_CLASS.get(el_type)
            if class_name is None:
                continue
            class_id = CLASS_TO_ID.get(class_name)
            if class_id is None:
                continue

            for idx, el in enumerate(elements):
                if (el_type, idx) in deleted or (el_type.rstrip("s"), idx) in deleted:
                    continue

                bbox = el.get("bbox", [])
                if len(bbox) < 4:
                    continue

                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                if not (0 < bw <= 1 and 0 < bh <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
                    continue

                labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        for split in splits_for_job:
            dest_lbl = dataset_dir / "labels" / split / f"{job_id}.txt"
            dest_lbl.write_text("\n".join(labels))
            print(f"  {job_id[:8]}… [{split}] — {len(labels)} labels, "
                  f"{len(deleted)} deleted")
        # Count UNIQUE images, not (image, split) pairs. With
        # duplicate_into_both this fix prevents the log lying about
        # dataset size and any future budgeting from inflating.
        valid_count += 1

    return valid_count, strategy


def main():
    parser = argparse.ArgumentParser(
        description="Retrain column detector on correction data from CorrectionsLogger."
    )
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--min-corrections", type=int,   default=10)
    parser.add_argument("--imgsz",           type=int,   default=1280,
                        help="Match the IMGSZ used by train.py (default: 1280)")
    parser.add_argument("--base-weights",    default="column_detect.pt",
                        help="Starting weights (default: column_detect.pt at project root)")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Build dataset only; skip model training")
    args = parser.parse_args()

    db_path = Path("data/corrections.db")
    if not db_path.exists():
        print(f"ERROR: {db_path} not found. Run the pipeline and make corrections first.")
        sys.exit(1)

    base_weights = Path(args.base_weights)
    if not base_weights.exists() and not args.dry_run:
        print(f"ERROR: base weights not found at {base_weights}")
        sys.exit(1)

    corrections = load_corrections(db_path)
    print(f"Loaded {len(corrections)} corrections from {db_path}")

    if len(corrections) < args.min_corrections:
        print(
            f"Only {len(corrections)} correction(s) found "
            f"(minimum: {args.min_corrections}). "
            "Make more corrections in the UI before retraining."
        )
        sys.exit(0)

    n_deleted = sum(1 for c in corrections if c["is_delete"])
    n_edits   = len(corrections) - n_deleted
    print(f"  Edits: {n_edits}   Deletions (false-positives): {n_deleted}")

    dataset_dir = Path("data/yolo_finetune")
    print(f"\nBuilding YOLO dataset at {dataset_dir}/ ...")
    n_valid, val_strategy = build_dataset(corrections, dataset_dir)
    print(f"  val split strategy: {val_strategy}")

    if n_valid == 0:
        print(
            "\nNo valid training samples found.\n"
            "Ensure the pipeline has run with checkpoint saving enabled "
            "(data/jobs/{job_id}/render.jpg and px_detections.json)."
        )
        sys.exit(1)

    yaml_path = write_data_yaml(dataset_dir, CLASS_NAMES)
    print(f"\nDataset complete: {n_valid} image(s).  Config: {yaml_path}")

    if args.dry_run:
        print("Dry-run mode — skipping training.")
        return

    print(f"\nStarting YOLO fine-tuning  ({args.epochs} epochs, imgsz={args.imgsz})...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Refresh hard-negative pool before training so any newly-logged FPs
    # are folded in as zero-label background tiles. Best-effort: a missing
    # pool script or empty DB does not fail the retrain.
    try:
        subprocess.run(
            [sys.executable, "scripts/hard_negative_pool.py"],
            check=False, cwd=str(Path.cwd()),
        )
    except Exception as e:
        print(f"  (hard-neg pool refresh skipped: {e})")

    model = YOLO(str(base_weights))
    train_results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="runs/detect",
        name="correction_feedback",
        exist_ok=True,
        verbose=True,
    )

    best = Path("runs/detect/correction_feedback/weights/best.pt")
    if best.exists():
        ts  = int(time.time())
        dst = Path(f"column_detect_ft_{ts}.pt")
        shutil.copy2(best, dst)
        print(f"\nFine-tuned weights saved: {dst}")
        print("To deploy, inspect on a real plan first, then:")
        print(f"  cp {dst} column_detect.pt")
        # Manual promotion gate (spec: feedback-loop / Manual promotion):
        # column_detect.pt MUST NOT be auto-overwritten. Print a guard
        # message so anyone reading the retrain output sees the contract.
        print("  (column_detect.pt is NOT auto-overwritten — promote manually.)")

        # Emit per-revision metrics + audit provenance.
        try:
            _write_metrics(model, train_results, args, len(corrections),
                           project_root=Path.cwd(),
                           val_split_strategy=val_strategy)
        except Exception as e:
            print(f"  ! metrics emission failed: {e}")
    else:
        print("\nTraining complete but best.pt not found — check runs/detect/correction_feedback/")


if __name__ == "__main__":
    main()
