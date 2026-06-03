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

from PIL import Image, ImageOps

# Real plans at 300 DPI exceed Pillow's default decompression-bomb ceiling
# (~89 Mpx). The plans are trusted local files, not adversarial uploads.
Image.MAX_IMAGE_PIXELS = None

INPUT_DPI_DEFAULT = 300

# Anchor all paths to the project root via __file__ so the script works
# regardless of the caller's CWD (cron, subprocess from another dir, etc.).
SCRIPTS_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT     = SCRIPTS_DIR.parent
DATA_ROOT        = PROJECT_ROOT / "data"
RAW_DRAWINGS_DIR = DATA_ROOT / "raw" / "drawings"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EXIF_ORIENTATION_TAG = 0x0112   # https://exiv2.org/tags.html


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


def _delete_stale_siblings(drawing_id: str, keep: Path) -> None:
    """Delete any other-extension rasters for the same drawing_id so a
    stale `<id>.png` from a prior PDF ingest cannot win a glob race
    against a fresh `<id>.jpg`. Meta files are preserved.
    """
    for stale in RAW_DRAWINGS_DIR.glob(f"{drawing_id}.*"):
        if stale == keep:
            continue
        if stale.name.endswith(".meta.json"):
            continue
        try:
            stale.unlink()
            print(f"  (deleted stale sibling: {stale.name})")
        except OSError as e:
            print(f"  WARNING: could not delete stale {stale.name}: {e}")


def ingest(source: Path, drawing_id: str, dpi: int) -> Path:
    RAW_DRAWINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_meta = RAW_DRAWINGS_DIR / f"{drawing_id}.meta.json"
    src_suffix = source.suffix.lower()

    if not src_suffix:
        print(f"ERROR: source '{source}' has no extension; rename it to "
              "include one of .pdf/.png/.jpg/.jpeg/.tif/.tiff/.bmp first.",
              file=sys.stderr)
        sys.exit(2)

    if src_suffix == ".pdf":
        # Rasterise the PDF page to PNG at the requested DPI. PDF input
        # is the only path that creates a NEW raster artefact; image
        # input is just copied (next branch).
        img = _rasterise_pdf(source, dpi)
        out_path = RAW_DRAWINGS_DIR / f"{drawing_id}.png"
        _delete_stale_siblings(drawing_id, keep=out_path)
        # optimize=True can take minutes on 100+ Mpx PNGs. The plans are
        # not redistributed, so we skip the recompression — faster ingest
        # for the cost of a slightly larger file on disk.
        img.save(out_path)
        used_dpi = dpi
        width, height = img.size
    elif src_suffix in IMAGE_SUFFIXES:
        out_path = RAW_DRAWINGS_DIR / f"{drawing_id}{src_suffix}"
        # Probe source to decide between fast-copy and re-encode. We need
        # to read three things: number of frames (multi-page TIFF warn),
        # EXIF orientation (must NOT silently pass through — see below),
        # and declared DPI (fallback to caller's --dpi).
        with Image.open(source) as probe:
            n_frames = getattr(probe, "n_frames", 1)
            try:
                orient = probe.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1
            except Exception:
                orient = 1
            declared_dpi = probe.info.get("dpi")
            src_width, src_height = probe.size

        if n_frames > 1:
            # Match the PDF branch's warning — previously silent for image input.
            print(f"NOTE: image has {n_frames} frames; using page 1 only.")

        # EXIF Orientation: a JPEG/TIFF with Orientation=6 (90° CW), 3
        # (180°), etc. looks rotated in viewers (they honour EXIF) but PIL
        # in our inference path does NOT auto-transpose. Verbatim-copy
        # would leave the model seeing the un-rotated frame while the
        # reviewer marks bboxes on the rotated frame → silent corruption
        # of every correction. When orientation != 1 we MUST re-encode
        # with the rotation baked in and the tag dropped. Same for
        # multi-page input — re-encode collapses to page 1 explicitly.
        needs_reencode = (orient != 1) or (n_frames > 1)
        _delete_stale_siblings(drawing_id, keep=out_path)
        if needs_reencode:
            img = ImageOps.exif_transpose(Image.open(source))
            img.save(out_path)
            width, height = img.size
            if orient != 1:
                print(f"  (EXIF orientation={orient} baked in; "
                      "image rotated to match viewer rendering)")
        else:
            # Fast path: byte-identical copy.
            shutil.copy2(source, out_path)
            # Drop restrictive source perms so other tools / users / Docker
            # containers running as a different uid can read the file.
            try:
                out_path.chmod(0o644)
            except OSError:
                pass
            width, height = src_width, src_height

        if (declared_dpi and isinstance(declared_dpi, tuple)
                and declared_dpi[0]):
            used_dpi = int(round(float(declared_dpi[0])))
        else:
            used_dpi = dpi
    else:
        print(f"ERROR: unsupported source suffix '{source.suffix}'",
              file=sys.stderr)
        sys.exit(2)

    meta = {
        "drawing_id":  drawing_id,
        "dpi":         used_dpi,
        "source":      str(source.resolve()),
        "ingested_as": str(out_path.relative_to(DATA_ROOT)),
        "size":        [width, height],
        "ingested_ts": time.time(),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Ingested {source.name} → {out_path}  "
          f"(DPI: {used_dpi}, {width}×{height})")
    return out_path


def resolve_drawing(drawing_id: str) -> tuple[Path, dict]:
    """Resolve the canonical raster path for a drawing_id via meta.json.

    Returns (raster_path, meta_dict). Raises FileNotFoundError with a
    clear hint if the drawing has not been ingested yet — telling the
    user EXACTLY which command to run to fix it.

    This is the ONE function every consumer (notebook, retrain,
    hard-negative pool) should use to find the raster for a drawing-id.
    Globbing by suffix is brittle (stale-cache race when re-ingesting
    from different formats) and the meta.json is the source of truth.
    """
    meta_path = RAW_DRAWINGS_DIR / f"{drawing_id}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No ingest record for drawing_id={drawing_id!r}. Run:\n"
            f"  python3 scripts/hitl.py ingest <plan> --drawing-id {drawing_id}"
        )
    meta = json.loads(meta_path.read_text())
    raster_path = DATA_ROOT / meta["ingested_as"]
    if not raster_path.exists():
        # Self-healing fallback: meta says the raster lives at X but X is
        # gone (user deleted it). Try a glob by stem and re-pick the
        # newest match — better than crashing if the meta is salvageable.
        siblings = [p for p in RAW_DRAWINGS_DIR.glob(f"{drawing_id}.*")
                    if not p.name.endswith(".meta.json")]
        if not siblings:
            raise FileNotFoundError(
                f"meta.json points at {raster_path} but no raster found. "
                f"Re-run: python3 scripts/hitl.py ingest <plan> "
                f"--drawing-id {drawing_id}"
            )
        raster_path = max(siblings, key=lambda p: p.stat().st_mtime)
    return raster_path, meta


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
