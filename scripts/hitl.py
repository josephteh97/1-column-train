r"""HITL workflow CLI — one entry point for the whole human-in-the-loop loop.

The HOT loop has three phases. This script gives you ONE command per phase.

WORKED EXAMPLE — reviewing TGCH-TD-S-200-L3-00 (the L3.jpg plan).
Each command is ONE LINE. Do not split with backslashes — `\ ` (backslash
followed by a space) is a literal-space argument in bash and will confuse
argparse:

    # Phase 1 — PREP (quote the path because it contains a space):
    python3 scripts/hitl.py ingest '/home/jiezhi/Documents/TGCH floor plan/L3.jpg' --drawing-id TGCH-TD-S-200-L3-00

    # Phase 2 — REVIEW (interactive web reviewer is its own package):
    column-review
    # Browser opens. Pick TGCH-TD-S-200-L3-00 from the file picker,
    # enter your reviewer id, then mark FPs with F/click and drag-add
    # missed columns. Autosave is on; close the tab when done.

    # check anytime:
    python3 scripts/hitl.py status

    # Phase 3 — RETRAIN (once status shows >=10 corrections):
    python3 scripts/hitl.py retrain --epochs 30

    # Then inspect data/metrics/<ts>.json + test on a real plan, and:
    cp column_detect_ft_<ts>.pt column_detect.pt

What each placeholder means:

    <plan>        Path to the PDF or image to review. Quote it if the path
                  contains spaces. Examples:
                      '/home/jiezhi/Documents/TGCH floor plan/L3.jpg'
                      /home/jiezhi/Documents/floor_plans/L5.pdf

    --drawing-id  Stable identifier for this drawing. Pick something
                  unique-per-floor that you'll recognise later — the same
                  id reused on the same plan groups all corrections.
                  Examples:
                      TGCH-TD-S-200-L3-00
                      project-A-level-5

    --epochs N    How many epochs to fine-tune. Default 30 is fine for
                  a first retrain; bump to 50+ if you have many
                  corrections (>100). Higher = longer training time.

    --dry-run     Build data/yolo_finetune/ but skip the actual training.
                  Use to sanity-check the dataset before committing GPU
                  time.

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

    # Resolve the canonical raster path for the user-facing instructions.
    # The notebook now consumes DRAWING_ID + meta.json, so prefer telling
    # the user the drawing-id (not the source path, which for a PDF input
    # would be a .pdf that PIL can't open).
    canonical = None
    try:
        sys.path.insert(0, str(HERE))
        from ingest_drawings import resolve_drawing
        canonical, _meta = resolve_drawing(args.drawing_id)
    except Exception:
        pass   # fall through to the still-correct DRAWING_ID hint below

    print()
    print("=" * 60)
    print("PREP DONE. Next:")
    print(f"  1. Launch the correction reviewer:")
    print(f"        column-review")
    print(f"     (then pick {args.drawing_id} in the file picker.)")
    if canonical is not None:
        print(f"     (canonical raster: {canonical})")
    print(f"  2. Mark FPs (F or click); drag-add missed columns.")
    print(f"     Autosave is on; close the browser when done.")
    print(f"  3. When ≥10 corrections total are recorded, run:")
    print(f"        python3 scripts/hitl.py retrain")
    print("=" * 60)
    return 0


def cmd_build_tiles(args: argparse.Namespace) -> int:
    """Phase 1b — (re)generate the DZI tile pyramid for an existing drawing.

    Idempotent: any pre-existing `_files/` tree under the drawing-id is
    wiped first, so re-running on a complete pyramid replaces it cleanly.
    Use this for drawings ingested before the DZI step landed, or after
    a corrupt-pyramid recovery.
    """
    sys.path.insert(0, str(HERE))
    from ingest_drawings import resolve_drawing, _write_dzi
    from PIL import Image
    try:
        raster_path, meta = resolve_drawing(args.drawing_id)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Building DZI tile pyramid for {args.drawing_id} "
          f"({meta.get('size', '?')})...")
    with Image.open(raster_path) as src_img:
        _write_dzi(src_img, args.drawing_id)
    print(f"Wrote data/raw/drawings/{args.drawing_id}.dzi "
          "and the matching _files/ tree.")
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
        print(f"      then review it with `column-review` (pick <drawing-id>).")
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
    p_ing.add_argument("--no-tiles", action="store_true",
                       help="Skip DZI tile-pyramid generation. The web "
                            "reviewer will refuse to open the drawing "
                            "until tiles are built via `hitl.py "
                            "build-tiles <id>`. Tiles add ~25-35%% disk.")
    p_ing.set_defaults(func=cmd_ingest)

    p_bt = sub.add_parser("build-tiles",
                          help="Phase 1b — (re)generate the DZI tile pyramid.")
    p_bt.add_argument("drawing_id",
                      help="Drawing identifier whose canonical raster is "
                           "already ingested. Wipes and rewrites the "
                           "DZI tree under data/raw/drawings/<id>_files/. "
                           "Adds ~25-35%% disk on top of the raster.")
    p_bt.set_defaults(func=cmd_build_tiles)

    # The `review` subcommand is gone — Phase 2 now lives in the
    # top-level `column-review` package (`pip install -e .` registers
    # the console_script). See README + CLAUDE.md.

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
