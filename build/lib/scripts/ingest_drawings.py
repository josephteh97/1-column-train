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
import math
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

# Microsoft Deep Zoom Image (DZI) tile-pyramid constants. Consumed by
# OpenSeadragon in scripts/correction_app — values are part of the
# public interface and must not drift.
DZI_TILE_SIZE   = 256
DZI_OVERLAP     = 1
DZI_JPEG_QUALITY = 80


def _write_dzi(image: Image.Image, drawing_id: str) -> Path:
    """Write a Microsoft Deep Zoom Image pyramid for `image`.

    Output layout (per the OpenSeadragon-compatible DZI v2008 spec):
        data/raw/drawings/<drawing_id>.dzi               — XML manifest
        data/raw/drawings/<drawing_id>_files/<L>/<col>_<row>.jpg

    Level L=0 is a 1x1 thumbnail; L=max is full resolution. Tiles are
    256 pixels per side with a 1-pixel overlap on every interior edge.
    Uses cascade-downsample (each level is built from the previous, not
    from the full base) so the total work is O(W*H), not O(levels*W*H).

    Idempotent: wipes any existing `<id>_files/` tree before writing so
    a partial pyramid from a crashed prior run cannot survive.
    """
    W, H = image.size
    if W < 1 or H < 1:
        raise ValueError(f"image has zero extent: {W}x{H}")
    max_level = max(0, int(math.ceil(math.log2(max(W, H, 2)))))
    out_dzi   = RAW_DRAWINGS_DIR / f"{drawing_id}.dzi"
    out_files = RAW_DRAWINGS_DIR / f"{drawing_id}_files"
    if out_files.exists():
        shutil.rmtree(out_files)
    out_files.mkdir(parents=True, exist_ok=True)

    # Cascade: keep the current-level image in hand, halve it to make
    # the next-coarser level. Total resize cost ≈ 2 × W × H pixels,
    # versus naïvely re-resizing the base for every level which is
    # max_level × W × H — order-of-magnitude faster on A0/300DPI.
    lvl_img = image.convert("RGB")
    for level in range(max_level, -1, -1):
        lvl_w, lvl_h = lvl_img.size
        lvl_dir = out_files / str(level)
        lvl_dir.mkdir(parents=True, exist_ok=True)
        n_cols = (lvl_w + DZI_TILE_SIZE - 1) // DZI_TILE_SIZE
        n_rows = (lvl_h + DZI_TILE_SIZE - 1) // DZI_TILE_SIZE
        for row in range(n_rows):
            for col in range(n_cols):
                left   = max(0,     col * DZI_TILE_SIZE - DZI_OVERLAP)
                top    = max(0,     row * DZI_TILE_SIZE - DZI_OVERLAP)
                right  = min(lvl_w, (col + 1) * DZI_TILE_SIZE + DZI_OVERLAP)
                bottom = min(lvl_h, (row + 1) * DZI_TILE_SIZE + DZI_OVERLAP)
                tile = lvl_img.crop((left, top, right, bottom))
                tile.save(lvl_dir / f"{col}_{row}.jpg",
                          "JPEG", quality=DZI_JPEG_QUALITY, optimize=False)
        if level == 0:
            break
        new_w = max(1, (lvl_w + 1) // 2)
        new_h = max(1, (lvl_h + 1) // 2)
        lvl_img = lvl_img.resize((new_w, new_h), Image.LANCZOS)

    out_dzi.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image TileSize="{DZI_TILE_SIZE}" Overlap="{DZI_OVERLAP}" '
        f'Format="jpg" '
        f'xmlns="http://schemas.microsoft.com/deepzoom/2008">\n'
        f'  <Size Width="{W}" Height="{H}"/>\n'
        '</Image>\n'
    )
    return out_dzi


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


def ingest(source: Path, drawing_id: str, dpi: int,
           build_tiles: bool = True) -> Path:
    RAW_DRAWINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_meta = RAW_DRAWINGS_DIR / f"{drawing_id}.meta.json"
    src_suffix = source.suffix.lower()
    # Hold a PIL.Image for the DZI step so we don't re-decode the
    # raster after writing it. None means "load from out_path".
    dzi_source_img: Image.Image | None = None

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
        dzi_source_img = img
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
            dzi_source_img = img
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

    if build_tiles:
        # DZI tile pyramid for the web correction viewer. Skipping this
        # (via --no-tiles) is only useful for ingest-only workflows;
        # `python3 scripts/hitl.py review <id>` refuses to open a
        # drawing whose DZI is missing.
        print(f"  building DZI tile pyramid (~25-35% additional disk)...",
              flush=True)
        if dzi_source_img is None:
            with Image.open(out_path) as src_img:
                _write_dzi(src_img, drawing_id)
        else:
            _write_dzi(dzi_source_img, drawing_id)
        print(f"  wrote DZI: data/raw/drawings/{drawing_id}.dzi")
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
    # Additive enrichment: report the DZI manifest path if it exists,
    # else None. Existing callers ignore unknown meta keys. Used by the
    # FastAPI correction-app to refuse a drawing whose tiles are missing.
    dzi_path = RAW_DRAWINGS_DIR / f"{drawing_id}.dzi"
    meta["dzi_path"] = str(dzi_path) if dzi_path.exists() else None
    return raster_path, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="Path to the PDF or image")
    p.add_argument("--drawing-id", default=None,
                   help="Drawing identifier (default: source filename stem)")
    p.add_argument("--dpi", type=int, default=INPUT_DPI_DEFAULT,
                   help=f"DPI for PDF rasterisation / image fallback (default: {INPUT_DPI_DEFAULT})")
    p.add_argument("--no-tiles", action="store_true",
                   help="Skip DZI tile-pyramid generation. The web "
                        "correction reviewer will refuse to open the "
                        "drawing until tiles are built via "
                        "`python3 scripts/hitl.py build-tiles <id>`. "
                        "DZI adds ~25-35%% disk on top of the raster.")
    args = p.parse_args()
    if not args.source.exists():
        print(f"ERROR: {args.source} not found", file=sys.stderr)
        sys.exit(1)
    drawing_id = args.drawing_id or args.source.stem
    ingest(args.source, drawing_id, args.dpi,
           build_tiles=not args.no_tiles)


if __name__ == "__main__":
    main()
