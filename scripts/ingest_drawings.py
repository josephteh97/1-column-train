"""Ingest a real floor-plan PDF or image into the data corpus.

Rasterises PDFs at a configurable DPI, copies images, and writes a
side-car JSON recording the DPI actually used. Output layout:

    data/raw/drawings/<drawing-id>.png        # rasterised plan
    data/raw/drawings/<drawing-id>.meta.json  # { "drawing_id", "dpi", "source", "ingested_ts" }

The `<drawing-id>` is either supplied by the caller (--drawing-id) or
derived from the source filename stem. DPI defaults to INPUT_DPI=300.

Usage:
    python3 scripts/ingest_drawings.py path/to/L3.pdf
    python3 scripts/ingest_drawings.py path/to/L3.jpg --drawing-id TGCH-TD-S-200-L3-00
    python3 scripts/ingest_drawings.py path/to/L3.pdf --dpi 300

PDF support requires pdf2image (and the system poppler-utils). Image
support requires only Pillow. Both are runtime-checked, not import-
time, so the script imports clean even without pdf2image.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

INPUT_DPI_DEFAULT = 300
RAW_DRAWINGS_DIR = Path("data/raw/drawings")


def _rasterise_pdf(pdf_path: Path, dpi: int) -> Image.Image:
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print(
            "ERROR: pdf2image not installed. Install with `pip install pdf2image`"
            " and ensure poppler-utils is on PATH.",
            file=sys.stderr,
        )
        sys.exit(2)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    if not pages:
        print(f"ERROR: pdf2image returned no pages for {pdf_path}", file=sys.stderr)
        sys.exit(2)
    if len(pages) > 1:
        print(
            f"NOTE: PDF has {len(pages)} pages; using page 1 only "
            "(structural drawings are typically one page).",
        )
    return pages[0].convert("RGB")


def _ingest_image(image_path: Path, fallback_dpi: int) -> tuple[Image.Image, int]:
    img = Image.open(image_path).convert("RGB")
    declared_dpi = img.info.get("dpi")
    if declared_dpi and isinstance(declared_dpi, tuple) and declared_dpi[0]:
        dpi = int(round(float(declared_dpi[0])))
    else:
        dpi = fallback_dpi
    return img, dpi


def ingest(source: Path, drawing_id: str, dpi: int) -> Path:
    RAW_DRAWINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = RAW_DRAWINGS_DIR / f"{drawing_id}.png"
    out_meta = RAW_DRAWINGS_DIR / f"{drawing_id}.meta.json"

    if source.suffix.lower() == ".pdf":
        img = _rasterise_pdf(source, dpi)
        used_dpi = dpi
    elif source.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        img, used_dpi = _ingest_image(source, dpi)
    else:
        print(f"ERROR: unsupported source suffix {source.suffix}", file=sys.stderr)
        sys.exit(2)

    img.save(out_png, optimize=True)
    meta = {
        "drawing_id": drawing_id,
        "dpi": used_dpi,
        "source": str(source.resolve()),
        "size": list(img.size),
        "ingested_ts": time.time(),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Ingested {source.name} → {out_png}  (DPI: {used_dpi}, {img.size[0]}×{img.size[1]})")
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="Path to the PDF or image")
    p.add_argument("--drawing-id", default=None,
                   help="Drawing identifier (default: source filename stem)")
    p.add_argument("--dpi", type=int, default=INPUT_DPI_DEFAULT,
                   help=f"DPI for PDF rasterisation / image fallback (default: {INPUT_DPI_DEFAULT})")
    args = p.parse_args()
    if not args.source.exists():
        print(f"ERROR: {args.source} not found", file=sys.stderr)
        sys.exit(1)
    drawing_id = args.drawing_id or args.source.stem
    ingest(args.source, drawing_id, args.dpi)


if __name__ == "__main__":
    main()
