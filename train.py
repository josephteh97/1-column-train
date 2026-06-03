"""
train.py – YOLOv11 structural-column detector (fine-tuned from COCO weights)
==================================================================

Learning approach
─────────────────
Supervised learning is the right paradigm here.  YOLO is a supervised
regression/classification model: every image has labels → the model learns
the mapping image → {bbox, class}.  This is exactly what the synthetic
generator produces.

Reinforcement learning is NOT suitable for object detection:
  RL optimises a policy through trial-and-error rewards, designed for
  sequential decision tasks (game agents, robotics).  Adapting it to
  detection adds enormous complexity with no accuracy benefit.

Recommended training roadmap
─────────────────────────────
  Phase 1 – columns only (this script)
    • Architecture : yolo11n-column.yaml (P2 head, nc=1, random init)
    • Output       : column_detect.pt

  Phase 2 – add door / wall / stairs
    • Architecture : same yaml but nc=4 (or a combined data.yaml with nc=4)
    • Weights init : column_detect.pt  ← backbone transfers, head re-inits
    • Example:
        model = YOLO("column_detect.pt")
        model.train(data="dataset_all.yaml", epochs=50, freeze=10)
      freeze=10 locks the first 10 backbone layers so only the upper
      features and detection head are updated initially.

  Long-term (recommended)
    • Once datasets for all 4 classes exist, do one joint training run
      with nc=4 starting from column_detect.pt.  A single model is simpler
      to deploy and the shared backbone is more efficient than four separate
      detectors.
"""

import shutil
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Select which class to train.  Each class lives in dataset/<CLASS>/ and must
# contain images/{train,val,test}/ + labels/{train,val,test}/ + data.yaml.
# Supported now: "column"
# Add later:     "door" | "wall" | "beam"
CLASS = "column"

# Model architecture.
# yolo11n.yaml  – standard P3/P4/P5 head, ~2.6 M params, fits 8 GB VRAM at
#                 imgsz=1280 batch=8.  No download needed (bundled in ultralytics).
# yolo11n-column.yaml – adds a P2 (stride-4) head for very small objects but
#                 needs ~12 GB VRAM at imgsz=1280.  Use only if you have 16 GB+.
MODEL_YAML  = "yolo11s.pt"                 # COCO-pretrained, ~9 M params. Pretrained
                                            # weights matter: 1300 synthetic tiles is
                                            # too few to train a non-tiny model from
                                            # scratch, which is why yolo11m.yaml
                                            # regressed on the real plan.

DATA_YAML   = f"dataset/{CLASS}/data.yaml" # per-class dataset config

EPOCHS      = 50
IMGSZ       = 1280      # MUST match TILE_SIZE in generate_column.py so training
                        # column pixel sizes equal inference column pixel sizes.
                        # Lowering to 1024 shrinks columns ~20 % and risks
                        # regressing the proven baseline.
BATCH       = 4         # BatchNorm needs ≥4; mosaic=0.5 keeps batch=4 in 8 GB VRAM.
WORKERS     = 2         # 4 workers caused pin_memory OOM cascade; 2 is safe
PATIENCE    = 15        # stop when best.pt hasn't improved for this many epochs

OUTPUT_NAME = f"{CLASS}_detect"             # best weights → column_detect.pt
RUN_NAME    = f"{CLASS}_detector"           # run folder name inside runs/detect/


# ── LEARNING CURVES ────────────────────────────────────────────────────────────
def plot_training_results(run_dir: Path) -> Path | None:
    """
    Read results.csv written by ultralytics during training and produce a
    clean 2-row figure:
      Row 1 – train / val losses  (box, class, DFL)
      Row 2 – validation metrics  (precision, recall, mAP50, mAP50-95)
    Saved as  <run_dir>/learning_curves.png
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"  [plot] results.csv not found in {run_dir}, skipping.")
        return None

    # Read CSV (column names have leading/trailing spaces in ultralytics output)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("  [plot] results.csv is empty, skipping.")
        return None

    # Normalise column names
    def col(row, key):
        for k in row:
            if k.strip() == key.strip():
                return float(row[k]) if row[k].strip() else 0.0
        return None

    keys = [k.strip() for k in rows[0].keys()]
    epochs = [int(col(r, "epoch")) + 1 for r in rows]

    def series(name):
        """Return list of floats for a column, or None if absent."""
        if name not in keys:
            return None
        vals = []
        for r in rows:
            v = col(r, name)
            vals.append(v if v is not None else float("nan"))
        return vals

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("Training Results – Column Detector", fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    BLUE   = "#2196F3"
    ORANGE = "#FF5722"
    GREEN  = "#4CAF50"

    # Row 0 – losses
    loss_specs = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Class Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]
    for col_idx, (tr_key, vl_key, title) in enumerate(loss_specs):
        ax = fig.add_subplot(gs[0, col_idx])
        tr = series(tr_key)
        vl = series(vl_key)
        if tr:
            ax.plot(epochs, tr, color=BLUE,   label="train", linewidth=1.5)
        if vl:
            ax.plot(epochs, vl, color=ORANGE, label="val",   linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 0 col 3 – mAP50 + mAP50-95 together
    ax_map = fig.add_subplot(gs[0, 3])
    m50   = series("metrics/mAP50(B)")
    m5095 = series("metrics/mAP50-95(B)")
    if m50:
        ax_map.plot(epochs, m50,   color=GREEN,  label="mAP50",    linewidth=1.5)
    if m5095:
        ax_map.plot(epochs, m5095, color=ORANGE, label="mAP50-95", linewidth=1.5)
    ax_map.set_title("mAP", fontsize=10)
    ax_map.set_xlabel("Epoch", fontsize=8)
    ax_map.set_ylim(0, 1)
    ax_map.legend(fontsize=7)
    ax_map.grid(True, alpha=0.3)

    # Row 1 – per-metric curves
    metric_specs = [
        ("metrics/precision(B)", "Precision", BLUE),
        ("metrics/recall(B)",    "Recall",    ORANGE),
        ("metrics/mAP50(B)",     "mAP50",     GREEN),
        ("metrics/mAP50-95(B)", "mAP50-95",  "#9C27B0"),
    ]
    for col_idx, (key, title, colour) in enumerate(metric_specs):
        ax = fig.add_subplot(gs[1, col_idx])
        s = series(key)
        if s:
            ax.plot(epochs, s, color=colour, linewidth=1.5)
            # Mark best epoch
            best_idx = s.index(max(v for v in s if v == v))   # ignore NaN
            ax.axvline(epochs[best_idx], color="gray", linestyle="--",
                       alpha=0.6, linewidth=1)
            ax.text(epochs[best_idx], max(v for v in s if v == v),
                    f" best\n ep{epochs[best_idx]}",
                    fontsize=6, color="gray", va="top")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    out = run_dir / "learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── EVALUATION HELPER ─────────────────────────────────────────────────────────
def evaluate(weights: Path, split: str, run_dir: Path) -> dict:
    """
    Run model.val() on the given split, save plots to
    <run_dir>/eval_<split>/, and return a metric dict.
    """
    model = YOLO(str(weights))
    metrics = model.val(
        data    = DATA_YAML,
        split   = split,
        imgsz   = IMGSZ,
        batch   = BATCH,
        plots   = True,          # ← confusion matrix, PR/F1/P/R curves
        project = str(run_dir),
        name    = f"eval_{split}",
        exist_ok= True,
        verbose = False,
    )
    return {
        "mAP50":     metrics.box.map50,
        "mAP50-95":  metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall":    metrics.box.mr,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # ── Build model from local YAML (no internet required) ───────────────────
    model = YOLO(MODEL_YAML)
    print(f"Model  : {MODEL_YAML}  (COCO-pretrained, fine-tuned)")
    print(f"Params : {sum(p.numel() for p in model.model.parameters()):,}")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        workers   = WORKERS,
        patience  = PATIENCE,   # early-stop when best.pt has converged
        name      = RUN_NAME,   # ultralytics controls project dir; we only set name
        exist_ok  = True,
        save      = True,
        plots     = True,       # auto-generates confusion_matrix, PR/F1/P/R curves
        verbose   = True,

        # ── Numerical stability ──────────────────────────────────────────────
        # The COCO-pretrained backbone + easy synthetic distribution converges
        # in 1 epoch. Default lr0=0.01 (auto-optim) is 10x too hot here and
        # combined with fp16 AMP it triggered NaN loss at epoch 4 on the prior
        # run. Disable AMP (eliminates fp16 overflow ceiling) and lower the
        # initial LR. Slower per-epoch but stable.
        amp       = False,
        lr0       = 1e-3,

        # ── Augmentation ─────────────────────────────────────────────────────
        # Floor plans are grayscale architectural drawings – suppress colour
        # augmentations that would add unrealistic appearance variation.
        hsv_h     = 0.0,    # no hue shift (grayscale images)
        hsv_s     = 0.1,    # minimal saturation jitter
        hsv_v     = 0.3,    # brightness variation (lighting variation)
        degrees   = 0.0,    # columns are axis-aligned – no rotation
        translate = 0.1,
        scale     = 0.4,
        shear     = 0.0,
        perspective = 0.0,
        flipud    = 0.1,
        fliplr    = 0.5,
        mosaic    = 0.5,    # see BATCH note
        mixup     = 0.0,
    )

    # ── Resolve the actual save directory ultralytics used ────────────────────
    # Never guess the path — read it directly from the trainer object so it
    # works regardless of what ultralytics prepends (e.g. runs/detect/).
    run_dir = Path(model.trainer.save_dir)
    print(f"\nRun saved to: {run_dir.resolve()}")

    # ── Copy best weights to column_detect.pt ─────────────────────────────────
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"
        print("Warning: best.pt not found, falling back to last.pt")

    dest = Path(f"{OUTPUT_NAME}.pt")
    if best.exists():
        shutil.copy(best, dest)
        print(f"\nBest weights → {dest.resolve()}")

    # ── Custom learning curves (from results.csv) ─────────────────────────────
    print("\n── Generating learning curves ──")
    curves_path = plot_training_results(run_dir)
    if curves_path:
        print(f"  Learning curves  → {curves_path}")

    # ── Val-split evaluation + plots ─────────────────────────────────────────
    print("\n── Validation split ──")
    val_m = evaluate(best, "val", run_dir)
    for k, v in val_m.items():
        print(f"  {k:<12}: {v:.4f}")

    # ── Test-split evaluation + plots ─────────────────────────────────────────
    print("\n── Test split ──")
    test_m = evaluate(best, "test", run_dir)
    for k, v in test_m.items():
        print(f"  {k:<12}: {v:.4f}")

    # ── Summary of all output files ───────────────────────────────────────────
    print(f"""
── Output files ──────────────────────────────────────────
  Weights (best)        → {dest.resolve()}

  Training plots ({run_dir}/):
    results.png           – ultralytics summary grid
    learning_curves.png   – custom loss + metric curves
    confusion_matrix.png  – val confusion matrix (from training)
    PR_curve.png          – precision-recall curve (val)
    F1_curve.png          – F1 score curve (val)

  Validation eval ({run_dir}/eval_val/):
    confusion_matrix.png
    confusion_matrix_normalized.png
    PR_curve.png · F1_curve.png · P_curve.png · R_curve.png

  Test eval ({run_dir}/eval_test/):
    confusion_matrix.png
    confusion_matrix_normalized.png
    PR_curve.png · F1_curve.png · P_curve.png · R_curve.png
──────────────────────────────────────────────────────────
""")

    # ── Next steps ────────────────────────────────────────────────────────────
    print("""Next steps
──────────
1. Generate more data (aim ~1 000 train images):
       edit NUM_IMAGES / START_INDEX in generate_column.py → python3 generate_column.py

2. Phase 2 – transfer to door / wall / stairs:
       model = YOLO("column_detect.pt")
       model.train(data="dataset_all.yaml", epochs=50, freeze=10)
""")


if __name__ == "__main__":
    main()
