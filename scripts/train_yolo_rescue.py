"""Train the rescue YOLO (`column_rescue.pt`, yolo11n) from
`yolo11n.pt` COCO init on the unified rescue_tiles/ pool plus the
synthetic dataset.

This is the ONLY training script in the HITL loop. `column_detect.pt`
is never touched here — promotion is exclusively to `column_rescue.pt`,
gated by the absorption check.

Flow:

  1. Auto-invoke `rescue_tile_pool.py` so the pool reflects the
     latest corrections before training reads it.

  2. Assemble a data.yaml that lists BOTH:
       - synthetic dataset images (`dataset/column/images/{train,val}`)
         if present — dense, clean labels
       - `data/rescue_tiles/images/` — sparse real-plan tiles with
         positives + FP-marked negatives
     Synthetic primary, rescue-tiles fine-tuning bias. The val split
     stays synthetic-only so the absorption gate is the actual
     correction-coverage signal.

  3. Train yolo11n from `yolo11n.pt` (COCO-pretrained, ~2.6M params,
     fits 8GB at imgsz=1280 batch=4). Mirror `train.py`'s
     architectural-drawing augmentation policy (no rotation /
     perspective / hue jitter — locked in CLAUDE.md).

  4. Save trained weights to `column_rescue_quarantine_<ts>.pt`
     at the project root — NOT `column_rescue.pt`. Promotion is
     gated.

  5. Run `scripts/absorption_gate.run_gate` against the quarantine
     weights. On pass: move to `column_rescue.pt`, write
     `column_rescue.meta.json`, exit 0. On fail: leave
     `column_rescue.pt` unchanged, write meta with `gate_failure`
     block, exit 2 (UI surfaces the diagnostic from meta.json).

Usage:
    python3 scripts/train_yolo_rescue.py
    python3 scripts/train_yolo_rescue.py --epochs 30 --batch 4
    python3 scripts/train_yolo_rescue.py --dry-run    # assemble only
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

DATASET_ROOT = _PROJECT_ROOT / "dataset" / "column"
RESCUE_POOL  = _PROJECT_ROOT / "data" / "rescue_tiles"
BASE_WEIGHTS = "yolo11n.pt"   # COCO-pretrained init, fetched by ultralytics

# Locked by CLAUDE.md.
TILE_SIZE = 1280


def check_prerequisites() -> list[dict]:
    """Return missing-prereq descriptors, or []. Single source of
    truth for the train-rescue endpoint's 412 preflight.

    Synthetic dataset is OPTIONAL. Rescue pool is OPTIONAL. AT LEAST
    ONE must exist — the trainer can run on synthetic-only OR
    rescue-only data, but cannot run on neither.
    """
    out: list[dict] = []
    has_synthetic = (DATASET_ROOT / "data.yaml").is_file()
    rescue_imgs = RESCUE_POOL / "images"
    has_rescue = rescue_imgs.is_dir() and any(rescue_imgs.glob("*.jpg"))
    if not has_synthetic and not has_rescue:
        out.append({
            "code": "no_training_data",
            "what": "no training data available (no synthetic "
                    "dataset AND no rescue_tiles)",
            "fix":  "Either run `python3 generate_column.py` to build "
                    "the synthetic dataset, or open a drawing in "
                    "column-review and make some corrections "
                    "(FN_ADDED / FP marks). Then run this script "
                    "again.",
        })
    return out


def _assemble_data_yaml(scratch_dir: Path) -> Path:
    """Write a data.yaml that unions synthetic + rescue tiles.

    ultralytics accepts list-form `train:` and `val:`, so we can name
    multiple roots without a flat-symlink scratch dir. Each entry is
    an absolute path to an images/ directory; ultralytics derives the
    labels/ sibling automatically.

    val stays synthetic-only when synthetic exists — that's the
    deterministic mAP signal. When ONLY rescue tiles exist, val falls
    back to a small held-out slice of the rescue pool.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    syn_train = DATASET_ROOT / "images" / "train"
    syn_val   = DATASET_ROOT / "images" / "val"
    rescue_imgs = RESCUE_POOL / "images"

    train_roots: list[str] = []
    val_roots:   list[str] = []
    if syn_train.is_dir():
        train_roots.append(str(syn_train))
    if syn_val.is_dir():
        val_roots.append(str(syn_val))
    if rescue_imgs.is_dir() and any(rescue_imgs.glob("*.jpg")):
        train_roots.append(str(rescue_imgs))

    if not train_roots:
        raise RuntimeError("no training data on disk — preflight should "
                           "have caught this")
    if not val_roots:
        # No synthetic val available. Pointing val at rescue_imgs (the
        # only other thing on disk) would leak every training tile into
        # the val split and silently report inflated mAP. Better to
        # fail loud and force the user to either run
        # `generate_column.py` (which writes a clean val split) or
        # accept that the absorption gate's per-correction IoU check
        # is their only validation signal — both options are honest.
        raise RuntimeError(
            "no validation data on disk. dataset/column/images/val/ is "
            "missing and rescue_tiles/ alone cannot serve as a val "
            "split (every tile is in train). Run `python3 "
            "generate_column.py` to bootstrap the synthetic val split, "
            "then re-run."
        )

    data_yaml = scratch_dir / "data.yaml"
    data_yaml.write_text(
        f"# Auto-generated by scripts/train_yolo_rescue.py — do not edit.\n"
        f"# Union of synthetic dataset + data/rescue_tiles/.\n"
        f"path: /\n"
        f"train:\n"
        + "".join(f"  - {p}\n" for p in train_roots)
        + f"val:\n"
        + "".join(f"  - {p}\n" for p in val_roots)
        + f"nc: 1\n"
        f"names: [\"column\"]\n"
    )
    return data_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--imgsz",    type=int,   default=TILE_SIZE)
    parser.add_argument("--patience", type=int,   default=10)
    parser.add_argument("--lr0",      type=float, default=5e-4)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Refresh pool + assemble data.yaml only; "
                             "skip the training subprocess and gate.")
    parser.add_argument("--tau-fn", type=float, default=0.5)
    parser.add_argument("--tau-fp", type=float, default=0.3)
    parser.add_argument("--device",  default=None,
                        help="'cpu' / 'cuda:0' / 'mps' (auto if omitted)")
    args = parser.parse_args()

    # ── Preflight (mirrors what /api/train-rescue calls) ──
    missing = check_prerequisites()
    if missing:
        print("\nERROR: cannot start rescue training — prerequisites "
              "missing:", file=sys.stderr)
        for m in missing:
            print(f"  • {m['what']}\n      fix: {m['fix']}",
                  file=sys.stderr)
        return 2

    t0 = time.perf_counter()

    # ── 1. Refresh rescue_tiles/ from corrections.db ──
    print("\n── Refreshing rescue_tiles/ pool from corrections.db ──",
          flush=True)
    rc = subprocess.call(
        [sys.executable, str(_SCRIPTS_DIR / "rescue_tile_pool.py")],
        cwd=str(_PROJECT_ROOT),
    )
    if rc != 0:
        print("WARNING: rescue_tile_pool.py exited non-zero — proceeding "
              "with stale pool", file=sys.stderr)

    # ── 2. Assemble data.yaml ──
    print("\n── Assembling data.yaml ──", flush=True)
    scratch_dir = _PROJECT_ROOT / "runs" / "detect" / "rescue_scratch"
    data_yaml = _assemble_data_yaml(scratch_dir)
    print(f"Wrote {data_yaml}")
    print(data_yaml.read_text())

    if args.dry_run:
        print("(dry-run — skipping training and gate.)")
        return 0

    # ── 3. Train yolo11n from yolo11n.pt ──
    print("\n── Training yolo11n ──", flush=True)
    from ultralytics import YOLO
    model = YOLO(BASE_WEIGHTS)
    print(f"Model  : {BASE_WEIGHTS}  (COCO-pretrained)")
    print(f"Params : {sum(p.numel() for p in model.model.parameters()):,}")

    run_name = f"column_rescue_{int(time.time())}"
    model.train(
        data       = str(data_yaml),
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        workers    = 4,
        patience   = args.patience,
        name       = run_name,
        exist_ok   = True,
        save       = True,
        plots      = True,
        verbose    = True,
        device     = args.device,
        amp        = False,
        lr0        = args.lr0,
        # Floor-plans-friendly augmentation (mirror train.py).
        hsv_h      = 0.0,
        hsv_s      = 0.1,
        hsv_v      = 0.3,
        degrees    = 0.0,
        translate  = 0.1,
        scale      = 0.4,
        shear      = 0.0,
        perspective= 0.0,
        flipud     = 0.1,
        fliplr     = 0.5,
        mosaic     = 0.5,
        mixup      = 0.0,
    )

    run_dir = Path(model.trainer.save_dir)
    print(f"\nRun saved to: {run_dir}")

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"
        print("Warning: best.pt not found, using last.pt", flush=True)
    if not best.exists():
        print("ERROR: no weights produced by training", file=sys.stderr)
        return 2

    # ── 4. Write to quarantine path ──
    quarantine = _PROJECT_ROOT / f"column_rescue_quarantine_{int(time.time())}.pt"
    shutil.copy(best, quarantine)
    print(f"\nQuarantine weights → {quarantine}", flush=True)

    # ── 5. Run absorption gate ──
    print("\n── Absorption gate ──", flush=True)
    from absorption_gate import run_gate
    gate_result = run_gate(quarantine, _PROJECT_ROOT,
                            tau_fn=args.tau_fn, tau_fp=args.tau_fp)
    elapsed = time.perf_counter() - t0
    print(json.dumps(gate_result, indent=2), flush=True)

    rescue_pt   = _PROJECT_ROOT / "column_rescue.pt"
    rescue_meta = _PROJECT_ROOT / "column_rescue.meta.json"

    if gate_result["gate_status"] == "passed":
        # Promote: move quarantine to canonical path, write meta.
        shutil.move(str(quarantine), str(rescue_pt))
        meta = {
            **gate_result,
            "target_model":    "column_rescue.pt",
            "epochs_trained":  int(args.epochs),
            "duration_seconds": round(elapsed, 1),
            "device":          args.device or "auto",
            "saved_ts":        time.time(),
        }
        rescue_meta.write_text(json.dumps(meta, indent=2))
        print(f"\nPromoted to {rescue_pt} ({elapsed:.1f}s).")
        print(f"Meta : {rescue_meta}")
        print("\nNext: restart column-review (or just hit /api/infer) — "
              "the new weights load via mtime-keyed cache.")
        return 0
    else:
        # Hold the quarantine in place; record the failure so the UI
        # can surface it. column_rescue.pt is NOT overwritten.
        meta = {
            **gate_result,
            "target_model":     "column_rescue_quarantine_*.pt",
            "quarantine_path":  str(quarantine),
            "epochs_trained":   int(args.epochs),
            "duration_seconds": round(elapsed, 1),
            "device":           args.device or "auto",
            "saved_ts":         time.time(),
        }
        rescue_meta.write_text(json.dumps(meta, indent=2))
        print(f"\nGATE FAILED — quarantine retained at {quarantine}.",
              file=sys.stderr)
        print(f"Diagnostic written to {rescue_meta}.", file=sys.stderr)
        print(f"\nFix: address the {gate_result['gate_failure']['summary']}, "
              "gather more corrections, then re-run.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
