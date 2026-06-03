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
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

# Real plans at 300 DPI exceed Pillow's default decompression-bomb ceiling
# (~89 Mpx). The plans are trusted local files, not adversarial uploads.
Image.MAX_IMAGE_PIXELS = None

INPUT_DPI_DEFAULT = 300
RAW_DRAWINGS_DIR = Path("data/raw/drawings")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


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
    out_meta = RAW_DRAWINGS_DIR / f"{drawing_id}.meta.json"
    src_suffix = source.suffix.lower()

    if src_suffix == ".pdf":
        # Rasterise the PDF page to PNG at the requested DPI. PDF input
        # is the only path that creates a NEW raster artefact; image
        # input is just copied (next branch).
        img = _rasterise_pdf(source, dpi)
        out_path = RAW_DRAWINGS_DIR / f"{drawing_id}.png"
        # optimize=True can take minutes on 100+ Mpx PNGs. The plans are
        # not redistributed, so we skip the recompression — faster ingest
        # for the cost of a slightly larger file on disk.
        img.save(out_path)
        used_dpi = dpi
        width, height = img.size
    elif src_suffix in IMAGE_SUFFIXES:
        # Image input — copy verbatim. No re-encode, no decompression-
        # bomb warning, no quality loss, no minute-long PNG optimise pass.
        # Drawing-id naming convention preserved by keeping the source
        # extension on the output.
        out_path = RAW_DRAWINGS_DIR / f"{drawing_id}{src_suffix}"
        shutil.copy2(source, out_path)
        # Read size/DPI without decoding the full image pixel array.
        with Image.open(out_path) as probe:
            width, height = probe.size
            declared_dpi = probe.info.get("dpi")
        if (declared_dpi and isinstance(declared_dpi, tuple)
                and declared_dpi[0]):
            used_dpi = int(round(float(declared_dpi[0])))
        else:
            used_dpi = dpi
    else:
        print(f"ERROR: unsupported source suffix {source.suffix}", file=sys.stderr)
        sys.exit(2)

    meta = {
        "drawing_id":  drawing_id,
        "dpi":         used_dpi,
        "source":      str(source.resolve()),
        "ingested_as": str(out_path.relative_to(RAW_DRAWINGS_DIR.parent.parent)),
        "size":        [width, height],
        "ingested_ts": time.time(),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Ingested {source.name} → {out_path}  "
          f"(DPI: {used_dpi}, {width}×{height})")
    return out_path


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
