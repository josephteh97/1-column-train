"""`column-review` CLI entry-point.

Resolves the project root (so `scripts/corrections_logger.py` and
`scripts/retrain_yolo.py` remain importable from any CWD), picks a free
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
        "--classifier-weights", default=None,
        help=(
            "Override the CNN classifier weights path. Defaults to "
            "<project>/column_classifier.pt. If the file is absent the "
            "two-stage filter is skipped and the pipeline runs YOLO-only."
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
    p.add_argument(
        "--images-dir", default=None,
        help=(
            "Folder of PNG/JPG floor plans to expose in the file "
            "picker's 'Local images' section. Images are loaded "
            "directly into OpenSeadragon (no DZI tile pyramid) — "
            "skips the `hitl.py ingest` step. Default: "
            "/home/jiezhi/Documents/PDF TGCH Floor Plan All if it "
            "exists, otherwise none."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    project_root = _resolve_project_root()
    # Make `scripts.corrections_logger`, `scripts.postprocess_pipeline`,
    # etc. importable regardless of where the CLI was invoked from.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Deferred imports so `--help` works without pulling FastAPI/uvicorn
    # (faster startup, lower failure surface for the --help path).
    import uvicorn
    from column_review.server import (
        create_app, open_browser_soon, pick_port,
    )

    port = pick_port(args.port)
    bound_url = f"http://{args.host}:{port}"

    # Resolve images_dir — explicit flag wins, else try the TGCH
    # folder, else None (the local-image section stays hidden).
    if args.images_dir:
        images_dir = Path(args.images_dir).expanduser().resolve()
    else:
        candidate = Path(
            "/home/jiezhi/Documents/PDF TGCH Floor Plan All")
        images_dir = candidate if candidate.is_dir() else None

    config = {
        "project_root": project_root,
        "db_path": Path(args.db_path) if args.db_path else None,
        "weights_path": Path(args.weights) if args.weights else None,
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
