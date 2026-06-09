"""`column-review` CLI entry-point.

Resolves the project root (so `scripts/corrections_logger.py` and
`scripts/train_bbox_classifier.py` remain importable from any CWD), picks a free
port, mounts the FastAPI app, schedules the auto-open browser tab, and
hands control to uvicorn. Prints the chosen URL to stdout on startup so
the reviewer can confirm where the server is reachable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Project root is the parent of the `column_review/` package directory.
# After `pip install -e .` this stays stable; after a wheel install the
# fallback below uses the CWD-derived root (PROJECT_ROOT env override).
_PACKAGE_ROOT = Path(__file__).resolve().parent
_FALLBACK_PROJECT_ROOT = _PACKAGE_ROOT.parent


def _resolve_project_root() -> Path:
    env = os.environ.get("COLUMN_REVIEW_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return _FALLBACK_PROJECT_ROOT


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="column-review",
        description=(
            "Single-command web reviewer for YOLO column detections. "
            "Starts a local FastAPI server, auto-picks a free port if the "
            "default is in use, and opens the UI in your default browser."
        ),
    )
    p.add_argument(
        "--port", type=int, default=8765,
        help="Default port (auto-picks next free port if busy). [8765]",
    )
    p.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address. Loopback by default. [127.0.0.1]",
    )
    p.add_argument(
        "--db-path", default=None,
        help=(
            "Override the corrections SQLite path. Defaults to "
            "<project>/data/corrections.db."
        ),
    )
    p.add_argument(
        "--weights", default=None,
        help=(
            "Override the YOLO weights path. Defaults to "
            "<project>/column_detect.pt."
        ),
    )
    p.add_argument(
        "--rescue-weights", default=None,
        help=(
            "Override the rescue YOLO weights path. Defaults to "
            "<project>/column_rescue.pt. If the file is absent the "
            "two-detector union step soft-fails to main-YOLO-only output."
        ),
    )
    p.add_argument(
        "--rescue-conf-threshold", type=float, default=0.4,
        help=(
            "Rescue YOLO confidence cutoff (0..1). Conservative default "
            "of 0.4 until the first absorption-gate-passed training "
            "cycle establishes the working calibration. [0.4]"
        ),
    )
    p.add_argument(
        "--classifier-weights", default=None,
        help=(
            "Override the CNN classifier weights path. Defaults to "
            "<project>/column_classifier.pt. If the file is absent the "
            "veto stage is skipped and the pipeline runs YOLO+rescue-only."
        ),
    )
    p.add_argument(
        "--classifier-threshold", type=float, default=0.5,
        help=(
            "Classifier probability cutoff (0..1). Lower = keep more "
            "candidates, higher = stricter filter. [0.5]"
        ),
    )
    p.add_argument(
        "--no-browser", action="store_true",
        help="Do not auto-open the browser tab.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    project_root = _resolve_project_root()
    # Make `scripts.corrections_logger`, `scripts.postprocess_pipeline`,
    # etc. importable regardless of where the CLI was invoked from.
    from column_review.path_bootstrap import ensure_on_path
    ensure_on_path(project_root)

    # Deferred imports so `--help` works without pulling FastAPI/uvicorn
    # (faster startup, lower failure surface for the --help path).
    import uvicorn
    from column_review.server import (
        create_app, open_browser_soon, pick_port,
    )

    port = pick_port(args.port)
    bound_url = f"http://{args.host}:{port}"

    # Picker watches a single hard-wired folder. No flag override —
    # the workflow ("drop image → click → start reviewing") requires
    # a stable, known location, and the curated retrain dataset is
    # the only source we want to surface. Missing folder is non-fatal:
    # warn once and hide the "Local images" picker section.
    images_dir = Path("~/Documents/retrain-dataset").expanduser()
    if not images_dir.is_dir():
        print(f"[warn] watched folder missing: {images_dir} "
              "— picker 'Local images' section will be empty.",
              flush=True)
        images_dir = None
    else:
        images_dir = images_dir.resolve()

    config = {
        "project_root": project_root,
        "db_path": Path(args.db_path) if args.db_path else None,
        "weights_path": Path(args.weights) if args.weights else None,
        "rescue_weights": (
            Path(args.rescue_weights)
            if args.rescue_weights else None
        ),
        "rescue_conf_threshold": float(args.rescue_conf_threshold),
        "classifier_weights": (
            Path(args.classifier_weights)
            if args.classifier_weights else None
        ),
        "classifier_threshold": float(args.classifier_threshold),
        "host": args.host,
        "port": port,
        "images_dir": images_dir,
    }
    app = create_app(config)

    print(f"column-review listening on {bound_url}", flush=True)

    if not args.no_browser:
        open_browser_soon(bound_url)

    uvicorn.run(app, host=args.host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
