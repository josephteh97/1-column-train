"""Sequential wrapper — train the CNN classifier, then the rescue YOLO.

Architecture C absorbs every reviewer correction into BOTH trainable
models. This script is the single entry point that the 🧠 Train Both
button spawns:

  1. `scripts/train_bbox_classifier.py` — ~30 s on GPU. Writes
     `column_classifier.pt` + `.meta.json` (with the
     `latest_correction_ts_per_job` map). Releases GPU memory on exit.
  2. `scripts/train_yolo_rescue.py` — ~20 min on GPU. Auto-refreshes
     the `data/rescue_tiles/` pool, trains yolo11n, runs the
     absorption gate, and on pass promotes to `column_rescue.pt`.

Why sequential, not parallel:
  - Atomic semantics: CNN failure aborts before rescue runs, so the
    ⌫ Clear absorption gate (which reads MIN of both `meta` files'
    `latest_correction_ts_per_job`) never observes half-promoted state.
  - 8 GB GPU contention is avoided — CNN trains tiny (~98 k params)
    and frees its weights before the rescue YOLO's 1280×1280 tiles
    saturate VRAM.

Exit code is 0 on full success, non-zero on any failure. The status
poller in `column_review/retrain_jobs.py::_poll_loop` only distinguishes
success from failure, so finer-grained stage codes would be decorative.
The stage that failed is unambiguous from the tee'd log.

Usage:
    python3 scripts/train_both.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
from column_review.path_bootstrap import ensure_on_path   # noqa: E402
ensure_on_path(_SCRIPTS_DIR)


def check_prerequisites() -> list[dict]:
    """Single source of truth for what Train Both needs to run.

    Returns the concatenation of both child scripts' prereq lists in
    the same shape they emit individually (`{code, what, fix}`). The
    `/api/train-both` route imports this so its 412 preflight matches
    what a CLI `python3 scripts/train_both.py` invocation would catch
    — no CLI/route divergence.
    """
    from train_bbox_classifier import (   # noqa: E402
        check_prerequisites as _classifier_pre,
    )
    from train_yolo_rescue import (   # noqa: E402
        check_prerequisites as _rescue_pre,
    )
    return list(_classifier_pre()) + list(_rescue_pre())


def main() -> int:
    missing = check_prerequisites()
    if missing:
        print("\nERROR: cannot start Train Both — prerequisites missing:",
              file=sys.stderr)
        for m in missing:
            print(f"  • {m['what']}\n      fix: {m['fix']}",
                  file=sys.stderr)
        return 2

    print("\n══ Stage 1/2: CNN classifier (~30 s) ══", flush=True)
    rc = subprocess.call(
        [sys.executable, str(_SCRIPTS_DIR / "train_bbox_classifier.py")],
        cwd=str(_PROJECT_ROOT),
    )
    if rc != 0:
        print(f"\nCNN classifier exited {rc} — aborting before rescue. "
              "Partial promotion is not allowed under Architecture C "
              "(the ⌫ Clear absorption gate would refuse anyway).",
              file=sys.stderr)
        return rc

    print("\n══ Stage 2/2: rescue YOLO (~20 min) ══", flush=True)
    rc = subprocess.call(
        [sys.executable, str(_SCRIPTS_DIR / "train_yolo_rescue.py")],
        cwd=str(_PROJECT_ROOT),
    )
    if rc != 0:
        print(f"\nRescue YOLO exited {rc}. The CNN classifier was "
              "promoted in stage 1, so the next ⌫ Clear will block "
              "on the rescue side until the next 🧠 Train Both run.",
              file=sys.stderr)
        return rc

    print("\n══ Train Both done — both models promoted ══", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
