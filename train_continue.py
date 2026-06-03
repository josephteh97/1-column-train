"""
train_continue.py — gentle fine-tune of column_detect.pt on a NEW dataset.

Use this when the old training dataset is gone and you only have new tiles.
Goal: teach the model new column geometry WITHOUT destroying what it already
knows. Four guards against catastrophic forgetting:

  1. lr0 = 0.0001  (100× lower than scratch training)
  2. epochs = 3    (very short — model is already converged)
  3. freeze = 15   (lock backbone + neck; only detection head adapts)
  4. mosaic = 0.0  (no image mixing — cleaner gradient signal)

Output goes to `column_detect_continued.pt`. DO NOT overwrite column_detect.pt
automatically — inspect the new weights on a real plan first, then promote
manually:
    cp column_detect_continued.pt column_detect.pt
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

from train import DATA_YAML, IMGSZ, BATCH, WORKERS, plot_training_results, evaluate

# ── CONFIG ────────────────────────────────────────────────────────────────────
SOURCE_WEIGHTS = "column_detect.pt"
EPOCHS         = 3
LR0            = 0.0001
FREEZE         = 15                          # backbone + neck; only detection head adapts
PATIENCE       = 3

RUN_NAME       = "column_detector_continue"
OUTPUT_NAME    = "column_detect_continued"


def main():
    model = YOLO(SOURCE_WEIGHTS)
    print(f"Continuing from: {SOURCE_WEIGHTS}")
    print(f"Params         : {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"Freeze         : first {FREEZE} layers (backbone + neck frozen; head adapts)")
    print()

    model.train(
        data           = DATA_YAML,
        epochs         = EPOCHS,
        imgsz          = IMGSZ,
        batch          = BATCH,
        workers        = WORKERS,
        patience       = PATIENCE,
        name           = RUN_NAME,
        exist_ok       = True,
        save           = True,
        plots          = True,
        verbose        = True,

        # ── Continue-training knobs ──────────────────────────────────────────
        lr0            = LR0,
        warmup_epochs  = 0,
        freeze         = FREEZE,

        # ── Augmentation (same as scratch) ───────────────────────────────────
        hsv_h     = 0.0,
        hsv_s     = 0.1,
        hsv_v     = 0.3,
        degrees   = 0.0,
        translate = 0.1,
        scale     = 0.4,
        shear     = 0.0,
        perspective = 0.0,
        flipud    = 0.1,
        fliplr    = 0.5,
        mosaic    = 0.0,
        mixup     = 0.0,
    )

    run_dir = Path(model.trainer.save_dir)
    print(f"\nRun saved to: {run_dir.resolve()}")

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"
        print("Warning: best.pt not found, falling back to last.pt")
    if not best.exists():
        raise FileNotFoundError(f"No weights produced by training run at {run_dir / 'weights'}")

    dest = Path(f"{OUTPUT_NAME}.pt")
    shutil.copy(best, dest)
    print(f"\nContinued weights → {dest.resolve()}")
    print("\nNOTE: column_detect.pt (baseline) was NOT overwritten.")
    print("      Test the new weights on a real plan first. To promote:")
    print(f"        cp {dest} column_detect.pt")

    print("\n── Learning curves ──")
    curves = plot_training_results(run_dir)
    if curves:
        print(f"  → {curves}")

    print("\n── Validation split ──")
    for k, v in evaluate(best, "val", run_dir).items():
        print(f"  {k:<12}: {v:.4f}")


if __name__ == "__main__":
    main()
