"""
finalize.py — run after Ctrl-C'ing train.py once mAP has converged.

Picks up best.pt from the run dir, copies it to column_detect.pt, regenerates
learning_curves.png from results.csv, and runs val+test evaluations with plots.
"""
import shutil
from pathlib import Path

from train import DATA_YAML, OUTPUT_NAME, RUN_NAME, evaluate, plot_training_results

run_dir = Path("runs/detect") / RUN_NAME
best    = run_dir / "weights" / "best.pt"
if not best.exists():
    best = run_dir / "weights" / "last.pt"
    print("Warning: best.pt not found, falling back to last.pt")

dest = Path(f"{OUTPUT_NAME}.pt")
shutil.copy(best, dest)
print(f"Best weights → {dest.resolve()}")

print("\n── Learning curves ──")
curves = plot_training_results(run_dir)
if curves:
    print(f"  → {curves}")

print("\n── Validation split ──")
for k, v in evaluate(best, "val", run_dir).items():
    print(f"  {k:<12}: {v:.4f}")

print("\n── Test split ──")
for k, v in evaluate(best, "test", run_dir).items():
    print(f"  {k:<12}: {v:.4f}")

print(f"\nDone. Plots in {run_dir.resolve()}/ and {run_dir.resolve()}/eval_{{val,test}}/")
