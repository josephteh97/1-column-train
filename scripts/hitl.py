"""HITL workflow CLI — one entry point for the whole human-in-the-loop loop.

The HOT loop has three phases. This script gives you ONE command per phase:

    1. PREP    python3 scripts/hitl.py ingest <plan> --drawing-id <id>
                  Rasterises the plan, writes the side-car, refreshes
                  per-drawing splits, prints what to do next.

    2. REVIEW  (interactive — open correct_detections.ipynb, run cells)
                  This is the human-in-the-loop part; cannot be a CLI.
                  Use `python3 scripts/hitl.py status` any time to see
                  how many corrections you've accumulated.

    3. RETRAIN python3 scripts/hitl.py retrain [--epochs N] [--dry-run]
                  Refreshes the FP→hard-negative pool, runs the fine-tune,
                  prints the metrics file path and the manual-cp line.

Other useful subcommands:
    status      — corrections-DB summary + next-step hint.

Each subcommand is a thin wrapper around the existing scripts/* tools, so
you can still drop down to the lower-level CLIs when debugging. The flow
itself lives here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _run(cmd: list[str], *, cwd: Path = ROOT, check: bool = True) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def cmd_ingest(args: argparse.Namespace) -> int:
    """Phase 1 — ingest a real plan and refresh splits."""
    plan = Path(args.plan).resolve()
    if not plan.exists():
        print(f"ERROR: {plan} not found", file=sys.stderr)
        return 1

    ingest_cmd = [sys.executable, str(HERE / "ingest_drawings.py"),
                  str(plan), "--drawing-id", args.drawing_id]
    if args.dpi:
        ingest_cmd += ["--dpi", str(args.dpi)]
    rc = _run(ingest_cmd)
    if rc != 0:
        return rc

    rc = _run([sys.executable, str(HERE / "split_drawings.py")])
    if rc != 0:
        return rc

    print()
    print("=" * 60)
    print("PREP DONE. Next:")
    print(f"  1. Open correct_detections.ipynb in Jupyter.")
    print(f"  2. Set IMAGE_PATH = '{plan}' (cell 2).")
    print(f"  3. Run cells 1-8 in order. Mark FPs / add missed columns.")
    print(f"  4. When done with ≥10 corrections total, run:")
    print(f"        python3 scripts/hitl.py retrain")
    print("=" * 60)
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    """Phase 3 — refresh pool, run fine-tune, surface metrics + promotion."""
    # 1. Refresh the FP → hard-negative pool from current corrections.db.
    rc = _run([sys.executable, str(HERE / "hard_negative_pool.py")], check=False)
    if rc != 0:
        print("  (hard-negative pool refresh returned non-zero — continuing.)")

    # 2. Run the fine-tune.
    cmd = [sys.executable, str(HERE / "retrain_yolo.py"),
           "--epochs", str(args.epochs),
           "--min-corrections", str(args.min_corrections)]
    if args.dry_run:
        cmd.append("--dry-run")
    rc = _run(cmd, check=False)
    if rc != 0:
        return rc

    print()
    print("=" * 60)
    print("RETRAIN DONE. Next:")
    print("  1. Inspect data/metrics/<timestamp>.json — check raw vs filtered")
    print("     regression on TGCH-TD-S-200-L3-00 against expected=440.")
    print("  2. Open test_column.ipynb, set WEIGHTS to the new "
          "column_detect_ft_<ts>.pt, run it on a real plan to eyeball.")
    print("  3. If satisfied, promote manually:")
    print("        cp column_detect_ft_<ts>.pt column_detect.pt")
    print("=" * 60)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show corrections-DB summary + a next-step hint."""
    sys.path.insert(0, str(HERE))
    from corrections_logger import summary
    s = summary()
    print("Corrections DB:")
    print(f"  jobs (drawings reviewed) : {s['jobs']}")
    print(f"  effective deletes (FPs)  : {s['deletes']}")
    print(f"  edits / adds (FNs+edits) : {s['edits_or_adds']}")
    print(f"  rescinded deletes        : {s.get('rescinded_deletes', 0)}"
          "  (delete-then-edit; auto-filtered)")
    print(f"  total effective rows     : {s['corrections']}")
    print()
    n = s["corrections"]
    threshold = 10
    if n == 0:
        print(f"Next: ingest a plan with `hitl ingest <plan> --drawing-id <id>`,")
        print(f"      then review it in correct_detections.ipynb.")
    elif n < threshold:
        print(f"Next: keep reviewing — you have {n}/{threshold} corrections.")
    else:
        print(f"Next: `python3 scripts/hitl.py retrain` "
              f"(you have {n} ≥ {threshold} corrections).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="hitl",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="phase", required=True)

    p_ing = sub.add_parser("ingest", help="Phase 1 — rasterise plan + refresh splits.")
    p_ing.add_argument("plan", help="Path to the PDF or image to ingest.")
    p_ing.add_argument("--drawing-id", required=True,
                       help="Stable drawing identifier (e.g. TGCH-TD-S-200-L3-00).")
    p_ing.add_argument("--dpi", type=int, default=None,
                       help="Override DPI (default: ingest_drawings.py's INPUT_DPI=300).")
    p_ing.set_defaults(func=cmd_ingest)

    p_re = sub.add_parser("retrain", help="Phase 3 — refresh hard-neg pool + fine-tune.")
    p_re.add_argument("--epochs", type=int, default=30)
    p_re.add_argument("--min-corrections", type=int, default=10)
    p_re.add_argument("--dry-run", action="store_true",
                      help="Build the dataset only; skip the training call.")
    p_re.set_defaults(func=cmd_retrain)

    p_st = sub.add_parser("status", help="Show corrections-DB summary + next-step hint.")
    p_st.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
