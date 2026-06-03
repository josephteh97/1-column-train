"""
generate_column.py — synthetic floor-plan column dataset generator.

Pipeline: generate large A0 14043×9929 px canvases, tile each into 1280×1280
patches matching the inference tiling window.  All column / wall / annotation
sizes are calibrated in tile-pixel units so training tiles look identical to
inference tiles.

Each canvas can contain:
- 7 column shape classes (square, rect, circle, combined, unshaded variants)
- axis bubbles + north-arrow triangle markers
- big multi-bay OPENINGS — thick-walled rectangles with a big diagonal X-cross
  inside + optional `H-D2L2` style label, with labelled true-positive columns
  at the outer corners. Opening interiors are pre-checked to be column-free.
- 3-parallel-wall STAIRS with a perpendicular closing wall + steps + zigzag
  break line + labelled flanking columns
- segmented-wall LIFT shafts — opening-style walled box with a big X-cross,
  one wall broken into door-bay segments, + labelled corner/edge columns
  flush against the wall outside
- internal partition walls, unlabelled cores, slab signs
- unlabelled decoys for FP suppression (label-only text at empty intersections,
  grid-crossing markers)

Nothing in the pipeline draws over a labelled column — every drawer either
predates the column draw, sits in bay interiors that exclude column positions,
or only paints at empty grid intersections.

Output:  dataset/column/{images,labels}/{train,val,test}/  +  data.yaml.
Tiles named `{image_id*10000+tile_idx:08d}.png`.
When --no-human-check is NOT passed, an annotated overlay copy of every tile
(red rectangles around every YOLO label) is also written to
dataset/column/human_check/{train,val,test}/.

CLI:
    python3 generate_column.py                         # defaults
    python3 generate_column.py --canvases 50           # smaller dataset
    python3 generate_column.py --clean                 # wipe dataset first
    python3 generate_column.py --no-human-check        # skip overlay copies
    python3 generate_column.py --help                  # full options
"""

import argparse
import functools
import math
import os
import random
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# Architecture: generate LARGE floor-plan images that match the real A0 drawing
# at ~300 DPI, then TILE each into 1280×1280 training patches — exactly what
# inference does.  This eliminates the context mismatch that caused the 30% miss
# rate: training tiles now look identical to inference tiles.
#
# Real floor plan: A0 landscape @ 1:400 @ 300 DPI → 14043×9929 px (full size).
# Columns (800 mm @ 1:400 @ 300 DPI): 800 / 25.4 × 300 / 400 ≈ 23.6 px target.
#
# Tile training: each 14043×9929 canvas → 13×10 = 130 tiles @ TILE_SIZE 1280
# with TILE_STEP 1080 (200 px overlap, matching inference). With NUM_IMAGES
# = 200 canvases that yields ~26 000 training tiles total. All element sizes
# are calibrated to TILE_SIZE so they look correct in tiles.
NUM_IMAGES   = 200       # A0 canvases; each → ~130 tiles ≈ 26 000 total tiles
IMG_WIDTH    = 14043    # A0 landscape @ 300 DPI: 1189 mm × (300/25.4) = 14043 px
IMG_HEIGHT   = 9929     # A0 landscape @ 300 DPI:  841 mm × (300/25.4) =  9929 px
OUTPUT_DIR   = "dataset/column"
START_INDEX  = 000

# If True, every tile that has at least one column label is also saved with
# red bounding-box overlays to OUTPUT_DIR/human_check/<split>/<fname>.jpg so a
# human can visually QA the labels. CLI: --no-human-check to disable.
HUMAN_CHECK  = True

# Training tile size — matches YOLO imgsz and the inference tiling window.
# ALL element sizes below are in TILE_SIZE-relative pixels so every tile
# (whether from training or inference) shows elements at the same scale.
TILE_SIZE    = 1280
TILE_STEP    = 1080     # 200 px overlap between tiles (same as inference)

# Dataset split ratios (must sum to 1.0; remainder goes to test)
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.20
# TEST_RATIO = 0.10 (implicit)

# Fraction of canvases generated as pure negatives — grid, bubbles, partitions
# and slab decoys are drawn, but NO columns are placed. These canvases teach FP
# suppression on cover / notes / schedule pages that show grids without columns.
NEGATIVE_RATIO = 0.10

# Column pixel size — calibrated to TILE_SIZE space, not canvas space.
# At 1:400, 300 DPI: 800 mm → 23.6 px; 1200 mm → 35.4 px; 1400 mm → 41.3 px.
# Square/rect range 20–34 px matches typical C2 800×800 columns. ✓
# Round columns are commonly larger (1130–1400 Ø) so they get a wider band.
COL_MIN_SIZE = 16   # was 20 — real plan has small C-types (H-C9 etc.) down to ~16 px
COL_MAX_SIZE = 34
ROUND_MIN_SIZE = 24
ROUND_MAX_SIZE = 42

# Probability that a column is drawn with a LIGHTER gray fill (190–220) instead
# of the default medium gray (130–185). Real L5 plans have many small filled
# squares rendered in pale gray that the previous training distribution missed.
LIGHT_FILL_PROB = 0.25

# Probability that a column is drawn at a SMALLER size band (12–20 px) than the
# default. Real L5 plans have small C2/H-RCB7 squares around 14-18 px in tile
# space; the existing 16-34 band under-represents the very small end.
SMALL_COL_PROB = 0.20
SMALL_COL_MIN_SIZE = 12
SMALL_COL_MAX_SIZE = 20

# Probability of drawing an extra compact lift/stair core with LABELED columns
# tucked into its corners (column body partially overlapping the thick wall
# slab).  Real plans show this exact pattern at every core / stair shaft and
# the model misses them because the original training distribution only ever
# placed columns in clear white space.
CORE_CORNER_PROB = 0.25

# Set True to draw red bounding boxes on TILES for visual QA.
DRAW_DEBUG_BOXES = False

# Padding added on each side of a column bbox when emitting YOLO labels.
# Also used by the canvas-wide overlap chokepoint so the stored rects match
# the rectangles YOLO ground-truth boxes occupy (otherwise adjacent labels
# can overlap by 2*pad even though rendered columns are clear).
LABEL_PAD = 1   # 1-px breathing margin around the column outline. Used by
                # `_yolo_label` (YOLO bbox slightly bigger than the shape so
                # the solid outline is fully covered) AND by `_padded_rect`
                # (placement spacing in col_rects). Single source of truth.

# Per-channel pixel-value threshold for the end-of-frame orphan-label scrub.
# Canvas background is randint(248, 255). The lightest column fill maxes at
# 220 (LIGHT_FILL_PROB branch in place_column → randint(190, 220)). Outlines
# are 0-25 (near-black). 235 sits cleanly above the lightest fill and far
# above outline ink, so any sample reading ≥ this on all channels means the
# column was covered by a background-coloured draw and the label is orphaned.
ORPHAN_BG_THRESHOLD = 235

# ── COLUMN GEOMETRY TYPES (human-review reference) ────────────────────────────
# The trained model has ONE YOLO class (0 = column). The internal `cls` IDs
# below only control GEOMETRY of the drawn shape — they're never seen by the
# model. They exist so the synthetic dataset covers every shape variant that
# appears on real architectural floor plans.
#
#  cls │ shape                      │ fill   │ typical px │ real-plan example
#  ────┼────────────────────────────┼────────┼────────────┼──────────────────────
#   0  │ square (filled)            │ gray   │ 16–34 px   │ C2 800×800
#   1  │ rect (filled)              │ gray   │ 16–40 px   │ C2 800×600
#   2  │ circle (filled)            │ gray   │ 24–42 px   │ 1130 CIS round
#   3  │ combined (nested shapes)   │ gray   │ scaled up  │ rect-in-circle, etc.
#   4  │ circle, unshaded           │ white  │ 24–42 px   │ outline-only 1130 Ø
#   5  │ square, unshaded           │ white  │ 16–34 px   │ outline-only small sq
#   6  │ rect, unshaded             │ white  │ 16–40 px   │ outline-only rect
#
# Implementation:
#   - shape primitives: `_draw_square` / `_draw_rect` / `_draw_circle`.
#   - unshaded variants: `_make_unshaded` wraps a primitive with white fill
#     and a thicker outline.
#   - dispatch table: `_CLS_TO_SHAPE` maps cls → shape primitive.
#   - cls 3 (combined): drawn by `place_column`; outer shape is the bottom
#     element from a `COMBINED_PAIRS` tuple, inner shape is the top.
#
# Class-weight distributions live inline at each placement site:
#   - main grid loop in `generate_image`:  [55,28,6,8,15,7,3] over cls 0..6
#   - `_pick_tp_class_and_size`:           [60,20,8,12]       over cls 0/1/2/4
#   - `maybe_draw_core_with_corner_columns`: [60,22,3,4,3,6,2] over cls 0..6
#
# All (bottom, top) permutations for combined columns (cls 3). Rule: bottom
# element is always the larger one — it sets the bbox. Dispatched by the
# combined branch of `place_column` (look for "cls == 3").
COMBINED_PAIRS = [
    ("square", "circle"),
    ("rect",   "circle"),
    ("circle", "square"),
    ("circle", "rect"),
    ("square", "rect"),
    ("rect",   "square"),
]

# ── TEXT POOLS for unlabelled-decoy text and structure-interior labels ────────
# Surviving callers: column-label decoys (`draw_label_only_decoys`), beam-label
# decoys, core/lift/stair interior labels, room/wall-thickness labels in the
# unlabelled-core drawer, and slab signs in bay centres.
COLUMN_LABEL_POOL = [
    # Standard C-type (square/rect) — most common in the real plan
    "C2\n800×800", "C2\n800×800", "C2\n800×800",
    "C1\n600×600", "C3\n900×900", "C4\n1000×1000",
    # H-C type variants — explicitly present in the 515-column floor plan
    "H-C2\n800×800", "H-C2\n800×800",
    "H-C1\n600×600", "H-C3\n900×900",
    "H-C4\n1000×1000",
]
BEAM_LABEL_H_POOL = [
    "H-RCB1 800×800 CIS", "H-RCB3 800×800 CIS",
    "RCB3 800×800",        "RCB1 800×800 CIS",
    "H-RCB2 800×300 CIS",
]
# Labels inside special structures (stairwells, cores, lift shafts).
STRUCT_INNER_POOL = ["300 CIS", "H-D2L2", "H-D2L3", "RCB2 800×1000", "300"]
STRUCT_SIDE_POOL  = ["RCB2 800×300", "RCB2 800×1000", "RCB2 800×300"]
ROOM_LABEL_POOL   = [
    "CORRIDOR", "STAIRCASE", "LIFT LOBBY", "WC", "OFFICE",
    "M&E ROOM", "STORE", "LOBBY", "PLANT ROOM", "RISER",
]
WALL_THICK_POOL   = ["200", "150", "250", "200 CW", "150 CW", "100"]
SLAB_SIGN_POOL    = ["L1 SLAB", "L2 SLAB", "RC SLAB 200 THK", "DROP", "DROP SLAB",
                     "150 THK SLAB", "200 THK", "S1", "S2", "TYP. SLAB"]
# Short tokens rendered at column-pixel sizes (12-30 px tall) as unlabelled
# decoys — teaches the model that small letters / vertical beam tags / column
# tags are NOT columns. Includes glyphs whose silhouette resembles a small
# outlined column (B/D/O/8) which the real-plan FP run mistook for cls 5/6.
COLUMN_MIMIC_TOKEN_POOL = [
    "B", "D", "O", "8", "H", "R", "Q", "0",
    "C2", "C1", "RS1", "RCB3", "H-C2", "T1", "T2",
    "RC", "CW", "CL",
]


# ── FONT LOADING ────────────────────────────────────────────────────────────────
# Cached: late drawers (draw_column_labels, draw_small_text_decoys) call these
# hundreds of times per canvas, mostly with the same handful of sizes. Pillow
# does NOT cache ImageFont.truetype across calls, and the candidate-path scan
# below stats every file on every miss. lru_cache collapses that to <16 calls.
@functools.lru_cache(maxsize=32)
def _load_font(size):
    """Bold font for axis bubbles."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@functools.lru_cache(maxsize=32)
def _load_regular_font(size):
    """Regular (non-bold) font for annotation text overlays."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _paste_rotated_text(base_img, x, y, text, font, fill_rgb, angle=90):
    """Render text rotated by angle degrees and paste onto base_img centred at (x, y)."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    try:
        bb = probe.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        tw = max(len(text) * 7, 20)
        th = 14
    if tw < 1 or th < 1:
        return
    pad = 2
    txt_img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (255, 255, 255, 0))
    td = ImageDraw.Draw(txt_img)
    r, g, b = fill_rgb
    td.text((pad, pad), text, fill=(r, g, b, 255), font=font)
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        resample = Image.BICUBIC
    rotated = txt_img.rotate(angle, expand=True, resample=resample)
    px = max(0, min(base_img.width  - rotated.width,  x - rotated.width  // 2))
    py = max(0, min(base_img.height - rotated.height, y - rotated.height // 2))
    base_img.paste(rotated, (px, py), mask=rotated)


# ── OUTPUT SETUP ───────────────────────────────────────────────────────────────
def create_dirs():
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)
        if HUMAN_CHECK:
            os.makedirs(os.path.join(OUTPUT_DIR, "human_check", split),
                        exist_ok=True)


def write_dataset_meta():
    # Resolve to an absolute path so train.py works regardless of CWD and so
    # OUTPUT_DIR can be either relative ("dataset/column") or absolute.
    abs_path = os.path.abspath(OUTPUT_DIR)
    yaml_text = (
        "# YOLOv11 – Structural Column Detection Dataset\n"
        f"path: {abs_path}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: column\n"
    )
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_text)
    with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
        f.write("column\n")


# ── DRAWING PRIMITIVES ─────────────────────────────────────────────────────────
def dashed_line(draw, x1, y1, x2, y2, dash=22, gap=14, color=(70, 70, 70), w=1):
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos, on = 0.0, True
    while pos < length:
        seg  = dash if on else gap
        npos = min(pos + seg, length)
        if on:
            draw.line(
                [(x1 + dx * pos,  y1 + dy * pos),
                 (x1 + dx * npos, y1 + dy * npos)],
                fill=color, width=w,
            )
        pos, on = npos, not on


def _chain_line(draw, x1, y1, x2, y2, color=(40, 40, 40), width=1,
                long_dash=20, short_dash=4, gap=7):
    """Centre-line (chain line): long-dash / gap / short-dash / gap, repeating —
    the engineering-drawing convention for an axis or a void marker. The dash
    rhythm is a fixed pixel cadence (deliberately NOT TILE_SIZE-relative, since
    it's a drawing convention, not an object size); all three lengths must be
    positive so the march always advances."""
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pattern = ((long_dash, True), (gap, False), (short_dash, True), (gap, False))
    pos, i = 0.0, 0
    while pos < length:
        seg, on = pattern[i % 4]
        npos = min(pos + seg, length)
        if on:
            draw.line(
                [(x1 + dx * pos,  y1 + dy * pos),
                 (x1 + dx * npos, y1 + dy * npos)],
                fill=color, width=width,
            )
        pos, i = npos, i + 1


# ── Column shape primitives ───────────────────────────────────────────────────
# Each primitive draws ONE shape variant centred at (cx, cy) and returns
# `(bbox, (drawn_w, drawn_h))`. See the geometry-type reference at module top.

def _draw_square(draw, cx, cy, s, fill, outline, lw):
    """cls 0 — filled square (most common: C2 800×800 etc.)."""
    h = s // 2
    b = [cx - h, cy - h, cx + h, cy + h]
    draw.rectangle(b, fill=fill, outline=outline, width=lw)
    return b, (s, s)


def _draw_rect(draw, cx, cy, s, fill, outline, lw):
    """cls 1 — filled rect. Aspect ratio 0.55-0.85 (matches real columns
    like 800×500 (0.625) or 800×600 (0.75)). Never elongated/thin: long-thin
    rects in real plans are beams or walls, NOT columns. Random orientation."""
    asp = random.uniform(0.55, 0.85)
    if random.random() > 0.5:
        w, h = s, max(4, int(s * asp))
    else:
        w, h = max(4, int(s * asp)), s
    b = [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]
    draw.rectangle(b, fill=fill, outline=outline, width=lw)
    return b, (w, h)


def _draw_circle(draw, cx, cy, s, fill, outline, lw):
    """cls 2 — filled circle (typical: 1130 CIS round)."""
    r = max(5, s // 2)
    b = [cx - r, cy - r, cx + r, cy + r]
    draw.ellipse(b, fill=fill, outline=outline, width=lw)
    return b, (2 * r, 2 * r)


def _make_unshaded(fn):
    """Adapt a filled primitive to its unshaded (white-interior) twin used by
    cls 4 / 5 / 6 — same outline, white fill, randomised stroke width 2-3 px.
    The 1-px option was tried and dropped: `_is_orphan_label` samples at
    (size/2 - 1) which is exactly 1 px INSIDE a 1-px outline, so all 5 sample
    pixels read paper-bg and the scrub silently dropped ~25 % of unshaded
    labels — the very FN the variable-stroke change was meant to reduce."""
    def _draw(draw, cx, cy, s, _fill, outline, lw):
        w = random.choice([2, 2, 2, 3])
        return fn(draw, cx, cy, s, (255, 255, 255), outline, w)
    return _draw


_draw_circle_unshaded = _make_unshaded(_draw_circle)   # cls 4
_draw_square_unshaded = _make_unshaded(_draw_square)   # cls 5
_draw_rect_unshaded   = _make_unshaded(_draw_rect)     # cls 6


_SHAPE = {
    "square":          _draw_square,            # cls 0
    "rect":            _draw_rect,              # cls 1
    "circle":          _draw_circle,            # cls 2
    "circle_unshaded": _draw_circle_unshaded,   # cls 4
    "square_unshaded": _draw_square_unshaded,   # cls 5
    "rect_unshaded":   _draw_rect_unshaded,     # cls 6
}

# Single-shape cls → shape primitive. cls 3 is intentionally absent — it's
# handled separately in place_column (nested outer + inner shapes).
_CLS_TO_SHAPE = {
    0: "square",
    1: "rect",
    2: "circle",
    4: "circle_unshaded",
    5: "square_unshaded",
    6: "rect_unshaded",
}


def _random_bay(v_lines, h_lines):
    """Pick a random (vi, hi) bay and return its (x1, x2, y1, y2) corners."""
    vi = random.randint(0, len(v_lines) - 2)
    hi = random.randint(0, len(h_lines) - 2)
    return v_lines[vi], v_lines[vi + 1], h_lines[hi], h_lines[hi + 1]


def _text_centered(draw, x, y, text, fill, font):
    """draw.text centred at (x, y); falls back when Pillow lacks anchor= support."""
    try:
        draw.text((x, y), text, fill=fill, font=font, anchor="mm")
    except TypeError:
        draw.text((x - len(text) * 4, y - 8), text, fill=fill, font=font)


def _draw_hollow_core(draw, sx1, sy1, sx2, sy2, wt, x_cross=True):
    """Four thick wall slabs around a hollow interior; used for lift / stair cores."""
    wall_fill = (random.randint(125, 175),) * 3
    for slab in (
        [sx1,      sy1,      sx2,      sy1 + wt],
        [sx1,      sy2 - wt, sx2,      sy2     ],
        [sx1,      sy1,      sx1 + wt, sy2     ],
        [sx2 - wt, sy1,      sx2,      sy2     ],
    ):
        draw.rectangle(slab, fill=wall_fill, outline=(10, 10, 10), width=1)
    inner = [sx1 + wt, sy1 + wt, sx2 - wt, sy2 - wt]
    bg = random.randint(245, 255)
    draw.rectangle(inner, fill=(bg, bg, bg))
    if x_cross:
        _draw_x_cross(draw, *inner)
    return inner


def _draw_x_cross(draw, x1, y1, x2, y2, color=(40, 40, 40), inset=2, width=1,
                  dashed=False):
    """Two diagonals forming an X across the rect — used for openings/voids.
    `dashed=True` draws each diagonal as a centre-line (chain line) instead of
    a solid stroke."""
    for ax, ay, bx, by in ((x1 + inset, y1 + inset, x2 - inset, y2 - inset),
                           (x2 - inset, y1 + inset, x1 + inset, y2 - inset)):
        if dashed:
            _chain_line(draw, ax, ay, bx, by, color=color, width=width)
        else:
            draw.line([(ax, ay), (bx, by)], fill=color, width=width)


# ── COLUMN PLACEMENT ───────────────────────────────────────────────────────────
def place_column(draw, cx, cy, cls, size):
    """Single chokepoint for every column draw in the dataset. Returns
    `(outer_bbox, yolo_class=0)` — every variant collapses to YOLO class 0.

    Cls dispatch:
      0, 1, 2, 4, 5, 6 → _CLS_TO_SHAPE → shape primitive (square / rect /
                          circle, filled or unshaded). See geometry-type
                          table at module top.
      3                → combined branch below: outer + nested inner shape,
                          (bottom, top) drawn from `COMBINED_PAIRS`.

    Fill is medium-dark gray (130-185) with a near-black outline, except the
    LIGHT_FILL_PROB branch which uses 190-220 pale gray to match the small
    filled squares observed on real L5 plans.
    """
    if random.random() < LIGHT_FILL_PROB:
        # Lighter pale-gray fill — matches the small filled squares observed
        # under TOWER D1 on L5 that the medium-gray training pool missed.
        fill = (random.randint(190, 220),) * 3
    else:
        fill = (random.randint(130, 185),) * 3   # medium-dark gray, high contrast
    outline = (random.randint(0, 25),)    * 3   # near-black outline

    shape_name = _CLS_TO_SHAPE.get(cls)
    if shape_name is not None:
        bbox, _ = _SHAPE[shape_name](draw, cx, cy, size, fill, outline, 2)
        return bbox, 0

    # ── cls == 3  (combined: nested shapes) ──────────────────────────────────
    bot_shape, top_shape = random.choice(COMBINED_PAIRS)
    # Round-outer combined columns (1130 Ø CIS) need a larger floor than
    # square-outer ones to match the real plan's top-row column scale.
    floor = ROUND_MAX_SIZE - 4 if bot_shape == "circle" else COL_MAX_SIZE
    size  = max(size, floor)
    thick = random.randint(2, 3)

    outer_bbox, (outer_w, outer_h) = _SHAPE[bot_shape](
        draw, cx, cy, size, fill, outline, 1
    )
    outer_min = min(outer_w, outer_h)

    if bot_shape == "circle" and top_shape in ("square", "rect"):
        max_inner = max(8, int(outer_min * 0.68))
    else:
        max_inner = outer_min

    top_size = int(size * random.uniform(0.50, 0.68))
    top_size = min(top_size, max_inner - 2 * thick)
    top_size = max(2 * thick + 4, top_size)

    if top_size < outer_min - thick:
        _SHAPE[top_shape](draw, cx, cy, top_size, fill, outline, thick)

    return outer_bbox, 0


# ── GRID GENERATION ────────────────────────────────────────────────────────────
def make_grid(w, h):
    """Return (v_lines, h_lines) as pixel coordinate lists.

    Target span = 248 px (= 8400 mm @ 1:400 @ 300 DPI).
    On 8192×5760 canvas with 20–36 column lines and 10–20 row lines the grid
    density matches the real 515-column A0 floor plan.  Tiles cut at 1280 px
    from this canvas show the same column density as the real plan tiles.
    """
    # Margins: large enough that axis bubbles sit cleanly in the first tile row/col
    mx = random.randint(300, 500)
    my = random.randint(300, 500)
    nv = random.randint(28, 44)   # 28–44 column lines  (real plan has 42)
    nh = random.randint(15, 26)   # 15–26 row lines      (real plan has ~22)
    
    if w - 2 * mx < 1000 or h-2 * my < 1000:
        # Regenerate margins or adjust line counts
        mx, my = 200, 200
        
    #if lo > hi:
        #print(f"DEBUG: lo={lo}, hi={hi}, base={base}, avail={avail}, n={n}")

    def _spacings(n, avail):
        # Target 248 px / span (8400 mm @ 1:400 @ 300 DPI)
        base = max(220, avail // n)
        lo   = max(190, base - 50)
        hi   = min(310, base + 60)
        
        # CRITICAL FIX: Ensure lo <= hi
        hi = max(lo, hi)
            
        sp   = [random.randint(lo, hi) for _ in range(n - 1)]
        while sum(sp) > avail - 60:
            sp = [max(170, int(s * 0.96)) for s in sp]
        return sp

    v_lines = [mx]
    for s in _spacings(nv, w - 2 * mx):
        v_lines.append(v_lines[-1] + s)

    h_lines = [my]
    for s in _spacings(nh, h - 2 * my):
        h_lines.append(h_lines[-1] + s)

    return v_lines, h_lines


# ── YOLO LABEL + canvas-wide column-overlap chokepoint ────────────────────────
def _yolo_label(bbox, cls, pad=LABEL_PAD):
    """Convert a shape bbox to a YOLO label string with a small `LABEL_PAD`-px
    margin on each side (1 px). The margin guarantees the bbox visually
    clears the column's solid outline at any stroke weight used by
    `_make_unshaded` (2 or 3 px)."""
    x1, y1, x2, y2 = bbox
    x1 = max(0,          x1 - pad)
    y1 = max(0,          y1 - pad)
    x2 = min(IMG_WIDTH,  x2 + pad)
    y2 = min(IMG_HEIGHT, y2 + pad)
    bx = (x1 + x2) / 2 / IMG_WIDTH
    by = (y1 + y2) / 2 / IMG_HEIGHT
    bw = (x2 - x1)     / IMG_WIDTH
    bh = (y2 - y1)     / IMG_HEIGHT
    return f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}"


def _padded_rect(bbox, pad=LABEL_PAD):
    return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)


def _bbox_overlaps_any(bbox, others) -> bool:
    """True if `bbox` (x1,y1,x2,y2) overlaps any rect in `others`. Touching
    edges are not an overlap."""
    return any(not (bbox[2] <= r[0] or bbox[0] >= r[2]
                    or bbox[3] <= r[1] or bbox[1] >= r[3])
               for r in others)


def _square_overlaps_cols(cx, cy, size, col_rects) -> bool:
    """True if a size×size column centred at (cx, cy), padded to its YOLO-label
    extent, would overlap any rect in `col_rects`. The shared 'is this column
    slot already taken' check used by every column placer."""
    half = size // 2
    cand = _padded_rect((cx - half, cy - half, cx + half, cy + half))
    return _bbox_overlaps_any(cand, col_rects)


def _label_to_pixels(lbl):
    """Parse a YOLO label string into pixel-space `(cls, cx, cy, bw, bh)` on
    the full IMG_WIDTH × IMG_HEIGHT canvas."""
    p = lbl.split()
    return (int(p[0]),
            float(p[1]) * IMG_WIDTH,
            float(p[2]) * IMG_HEIGHT,
            float(p[3]) * IMG_WIDTH,
            float(p[4]) * IMG_HEIGHT)


def _all_background(pixels, points):
    """True if every (x, y) in `points` is paper-background — all channels
    >= ORPHAN_BG_THRESHOLD. Out-of-canvas points are skipped, not failed.
    Shared sampling primitive for `_is_orphan_label` and `_spot_is_clear`."""
    for sx, sy in points:
        ix, iy = int(sx), int(sy)
        if not (0 <= ix < IMG_WIDTH and 0 <= iy < IMG_HEIGHT):
            continue
        r, g, b = pixels[ix, iy]
        if not (r >= ORPHAN_BG_THRESHOLD
                and g >= ORPHAN_BG_THRESHOLD
                and b >= ORPHAN_BG_THRESHOLD):
            return False
    return True


def _is_orphan_label(lbl, pixels):
    """True if every sampled pixel inside the column bbox is paper-background
    — i.e. the column was drawn but a later draw (stair / lift / core X-cross
    interior, slab-sign text card, …) refilled it with bg.

    Five samples: centre + 4 outline mid-points just inside the actual column
    edge. This rule preserves UNSHADED outline columns (cls 5/6) whose
    interior IS background — at least one of the four outline samples lands
    on the dark outline pixel. A partially-covered column (e.g. half under a
    wall slab) still keeps at least one visible-column sample dark, so its
    label is preserved per the "kiss but not block" rule.

    Threshold derivation: see `ORPHAN_BG_THRESHOLD` at module top."""
    _, cx, cy, bw, bh = _label_to_pixels(lbl)
    # Actual column half-extent = YOLO half - LABEL_PAD. Step inward by 1 so
    # the sample lands ON the outline rather than just outside it.
    hx = bw / 2 - LABEL_PAD - 1
    hy = bh / 2 - LABEL_PAD - 1
    return _all_background(pixels, (
        (cx, cy),
        (cx - hx, cy), (cx + hx, cy),
        (cx, cy - hy), (cx, cy + hy)))


def _spot_is_clear(pixels, cx, cy, size):
    """True if a size×size column footprint centred at (cx, cy) lands on clear
    paper — nothing (wall slab, structure, X-cross, or an already-placed
    column) occupies it. Used to gate grid-column placement so a column is
    NEVER drawn over / under another element.

    Samples the 4 quadrant mid-points at offset `q = max(6, size//4)`. The
    dashed grid lines cross at the exact intersection centre; q ≥ 6 keeps
    every sample clear of those lines even at the ±3 px grid jitter
    (`randint(-3, 3)`), so a thin grid line never falsely reads as
    "occupied". A wall or structure fills the whole footprint and trips at
    least one sample."""
    q = max(6, size // 4)
    return _all_background(pixels, (
        (cx - q, cy - q), (cx + q, cy - q),
        (cx - q, cy + q), (cx + q, cy + q)))


# ── REVIT-STYLE GRID BUBBLE ────────────────────────────────────────────────────
def draw_bubble(draw, cx, cy, label, br, color, font, lw):
    draw.ellipse(
        [cx - br, cy - br, cx + br, cy + br],
        fill=(255, 255, 255),
        outline=color,
        width=lw,
    )
    try:
        draw.text((cx, cy), label, fill=color, font=font, anchor="mm")
    except TypeError:
        try:
            bb = draw.textbbox((0, 0), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            tw, th = len(label) * (br // 2), br
        draw.text((cx - tw // 2, cy - th // 2), label, fill=color, font=font)


# ── ANNOTATION OVERLAY ─────────────────────────────────────────────────────────
# ── SPECIAL STRUCTURE (stairwell / core wall / lift shaft) ─────────────────────
def maybe_draw_special_structure(draw, v_lines, h_lines, gc, ann_font, col_rects):
    """
    Draw a thick-WALLED hollow rectangle in ~65 % of images, mimicking the
    H-D2L2 lift core / stairwell visible in the real floor plan.

    Critical detail: the WALLS are drawn as solid gray-filled slabs, not just
    a thick outline.  This matches the real plan where each wall of the lift
    shaft appears as a gray rectangle — the exact false positive we saw.
    Training images must show these wall slabs as unlabeled background elements
    so the model learns: elongated gray slab ≠ column.
    """
    if random.random() > 0.65 or len(v_lines) < 2 or len(h_lines) < 2:
        return

    vi = random.randint(0, len(v_lines) - 2)
    hi = random.randint(0, len(h_lines) - 2)

    x1, x2 = v_lines[vi], v_lines[vi + 1]
    y1, y2 = h_lines[hi], h_lines[hi + 1]

    fw = int((x2 - x1) * random.uniform(0.30, 0.62))
    fh = int((y2 - y1) * random.uniform(0.30, 0.62))
    cx_s = (x1 + x2) // 2
    cy_s = (y1 + y2) // 2
    sx1, sy1 = cx_s - fw // 2, cy_s - fh // 2
    sx2, sy2 = cx_s + fw // 2, cy_s + fh // 2

    # Wall thickness: COL_MIN_SIZE / 3 to COL_MIN_SIZE / 2, making the slab
    # look like a real RC shear wall (same gray as columns but ELONGATED).
    wt = random.randint(COL_MIN_SIZE // 3, COL_MIN_SIZE // 2)
    _draw_hollow_core(draw, sx1, sy1, sx2, sy2, wt)

    # Interior annotation text ("H-D2L2", "300 CIS", etc.)
    lbl = random.choice(STRUCT_INNER_POOL)
    try:
        draw.text((cx_s, cy_s - 8), lbl, fill=gc, font=ann_font, anchor="mm")
    except TypeError:
        draw.text((cx_s - 30, cy_s - 12), lbl, fill=gc, font=ann_font)

    # Side labels and repeated "300 CIS" annotations
    for side_lbl, tx, ty in [
        (random.choice(STRUCT_SIDE_POOL), sx1 + 4, sy2 + 4),
        (random.choice(["300 CIS", "RCB2 800×300"]), cx_s, sy2 + 14),
    ]:
        if random.random() > 0.45:
            try:
                draw.text((tx, ty), side_lbl, fill=gc, font=ann_font)
            except Exception:
                pass

    # ── Door opening(s) (DO) ─────────────────────────────────────────────
    # 70 % of lift shafts have a door opening on one wall.
    # 50 % of those use DOUBLE openings: two gaps with a gray pier stub
    # between them.  The pier stub looks like a small column but is part of
    # the wall — the model must learn to ignore it.
    # Notation: gap in wall + straight line (panel at open) + quarter-arc.
    if fw > wt * 5 and fh > wt * 3 and random.random() < 0.70:
        door_wall = random.choice(["top", "bottom", "left", "right"])
        double    = random.random() < 0.50

        if door_wall == "top":
            span = fw - wt * 2
            if double and span > wt * 7:
                pw    = random.randint(wt, wt * 2)
                do_w  = max(wt * 2, (span - pw) // 2)
                total = do_w * 2 + pw
                bx    = max(sx1 + wt + 1, min(cx_s - total // 2, sx2 - wt - total - 1))
                gx1a  = bx
                gx2a  = bx + do_w
                gx1b  = bx + do_w + pw
                gx2b  = bx + do_w * 2 + pw
                hy_b  = sy1 + wt
                draw.rectangle([gx1a, sy1, gx2a, hy_b], fill=(252, 252, 252))
                draw.rectangle([gx1b, sy1, gx2b, hy_b], fill=(252, 252, 252))
                # Door 1: hinge at left of gap1, panel down, arc 0→90
                draw.line([(gx1a, hy_b), (gx1a, hy_b + do_w)], fill=(25, 25, 25), width=1)
                draw.arc([gx1a - do_w, hy_b - do_w, gx1a + do_w, hy_b + do_w],
                         start=0, end=90, fill=(25, 25, 25), width=1)
                # Door 2: hinge at right of gap2, panel down, arc 90→180
                draw.line([(gx2b, hy_b), (gx2b, hy_b + do_w)], fill=(25, 25, 25), width=1)
                draw.arc([gx2b - do_w, hy_b - do_w, gx2b + do_w, hy_b + do_w],
                         start=90, end=180, fill=(25, 25, 25), width=1)
                do_tx, do_ty = (gx1a + gx2b) // 2, sy1 - 10
            else:
                do_w = max(wt * 2 + 4, min(fw // 3, fw - wt * 4))
                offset = random.randint(-(fw // 6), fw // 6)
                gx1 = max(sx1 + wt + 2, cx_s + offset - do_w // 2)
                gx2 = min(sx2 - wt - 2, gx1 + do_w);  gx1 = gx2 - do_w
                draw.rectangle([gx1, sy1, gx2, sy1 + wt], fill=(252, 252, 252))
                hx, hy = gx1, sy1 + wt
                draw.line([(hx, hy), (hx, hy + do_w)], fill=(25, 25, 25), width=1)
                draw.arc([hx - do_w, hy - do_w, hx + do_w, hy + do_w],
                         start=0, end=90, fill=(25, 25, 25), width=1)
                do_tx, do_ty = (gx1 + gx2) // 2, sy1 - 10

        elif door_wall == "bottom":
            span = fw - wt * 2          # usable width between the two side wall slabs
            if double and span > wt * 7:
                pw    = random.randint(wt, wt * 2)          # pier width (column-stub)
                do_w  = max(wt * 2, (span - pw) // 2)
                total = do_w * 2 + pw
                bx    = max(sx1 + wt + 1, min(cx_s - total // 2, sx2 - wt - total - 1))
                gx1a  = bx
                gx2a  = bx + do_w
                gx1b  = bx + do_w + pw
                gx2b  = bx + do_w * 2 + pw
                hy_t  = sy2 - wt
                draw.rectangle([gx1a, hy_t, gx2a, sy2], fill=(252, 252, 252))
                draw.rectangle([gx1b, hy_t, gx2b, sy2], fill=(252, 252, 252))
                # Door 1: hinge at left of gap1, panel up, arc 270→360
                draw.line([(gx1a, hy_t), (gx1a, hy_t - do_w)], fill=(25, 25, 25), width=1)
                draw.arc([gx1a - do_w, hy_t - do_w, gx1a + do_w, hy_t + do_w],
                         start=270, end=360, fill=(25, 25, 25), width=1)
                # Door 2: hinge at right of gap2, panel up, arc 180→270
                draw.line([(gx2b, hy_t), (gx2b, hy_t - do_w)], fill=(25, 25, 25), width=1)
                draw.arc([gx2b - do_w, hy_t - do_w, gx2b + do_w, hy_t + do_w],
                         start=180, end=270, fill=(25, 25, 25), width=1)
                do_tx, do_ty = (gx1a + gx2b) // 2, sy2 + 10
            else:
                do_w = max(wt * 2 + 4, min(fw // 3, fw - wt * 4))
                offset = random.randint(-(fw // 6), fw // 6)
                gx1 = max(sx1 + wt + 2, cx_s + offset - do_w // 2)
                gx2 = min(sx2 - wt - 2, gx1 + do_w);  gx1 = gx2 - do_w
                draw.rectangle([gx1, sy2 - wt, gx2, sy2], fill=(252, 252, 252))
                hx, hy = gx1, sy2 - wt
                draw.line([(hx, hy), (hx, hy - do_w)], fill=(25, 25, 25), width=1)
                draw.arc([hx - do_w, hy - do_w, hx + do_w, hy + do_w],
                         start=270, end=360, fill=(25, 25, 25), width=1)
                do_tx, do_ty = (gx1 + gx2) // 2, sy2 + 10

        elif door_wall == "left":
            span = fh - wt * 2
            if double and span > wt * 7:
                pw    = random.randint(wt, wt * 2)
                do_w  = max(wt * 2, (span - pw) // 2)
                total = do_w * 2 + pw
                by    = max(sy1 + wt + 1, min(cy_s - total // 2, sy2 - wt - total - 1))
                gy1a  = by
                gy2a  = by + do_w
                gy1b  = by + do_w + pw
                gy2b  = by + do_w * 2 + pw
                hxi   = sx1 + wt
                draw.rectangle([sx1, gy1a, hxi, gy2a], fill=(252, 252, 252))
                draw.rectangle([sx1, gy1b, hxi, gy2b], fill=(252, 252, 252))
                # Door 1: hinge at top of gap1, panel right, arc 0→90
                draw.line([(hxi, gy1a), (hxi + do_w, gy1a)], fill=(25, 25, 25), width=1)
                draw.arc([hxi - do_w, gy1a - do_w, hxi + do_w, gy1a + do_w],
                         start=0, end=90, fill=(25, 25, 25), width=1)
                # Door 2: hinge at bottom of gap2, panel right, arc 270→360
                draw.line([(hxi, gy2b), (hxi + do_w, gy2b)], fill=(25, 25, 25), width=1)
                draw.arc([hxi - do_w, gy2b - do_w, hxi + do_w, gy2b + do_w],
                         start=270, end=360, fill=(25, 25, 25), width=1)
                do_tx, do_ty = sx1 - 10, (gy1a + gy2b) // 2
            else:
                do_w = max(wt * 2 + 4, min(fw // 3, fh - wt * 4))
                offset = random.randint(-(fh // 6), fh // 6)
                gy1 = max(sy1 + wt + 2, cy_s + offset - do_w // 2)
                gy2 = min(sy2 - wt - 2, gy1 + do_w);  gy1 = gy2 - do_w
                draw.rectangle([sx1, gy1, sx1 + wt, gy2], fill=(252, 252, 252))
                hx, hy = sx1 + wt, gy1
                draw.line([(hx, hy), (hx + do_w, hy)], fill=(25, 25, 25), width=1)
                draw.arc([hx - do_w, hy - do_w, hx + do_w, hy + do_w],
                         start=0, end=90, fill=(25, 25, 25), width=1)
                do_tx, do_ty = sx1 - 10, (gy1 + gy2) // 2

        else:  # right wall
            span = fh - wt * 2
            if double and span > wt * 7:
                pw    = random.randint(wt, wt * 2)
                do_w  = max(wt * 2, (span - pw) // 2)
                total = do_w * 2 + pw
                by    = max(sy1 + wt + 1, min(cy_s - total // 2, sy2 - wt - total - 1))
                gy1a  = by
                gy2a  = by + do_w
                gy1b  = by + do_w + pw
                gy2b  = by + do_w * 2 + pw
                hxi   = sx2 - wt
                draw.rectangle([hxi, gy1a, sx2, gy2a], fill=(252, 252, 252))
                draw.rectangle([hxi, gy1b, sx2, gy2b], fill=(252, 252, 252))
                # Door 1: hinge at top of gap1, panel left, arc 90→180
                draw.line([(hxi, gy1a), (hxi - do_w, gy1a)], fill=(25, 25, 25), width=1)
                draw.arc([hxi - do_w, gy1a - do_w, hxi + do_w, gy1a + do_w],
                         start=90, end=180, fill=(25, 25, 25), width=1)
                # Door 2: hinge at bottom of gap2, panel left, arc 180→270
                draw.line([(hxi, gy2b), (hxi - do_w, gy2b)], fill=(25, 25, 25), width=1)
                draw.arc([hxi - do_w, gy2b - do_w, hxi + do_w, gy2b + do_w],
                         start=180, end=270, fill=(25, 25, 25), width=1)
                do_tx, do_ty = sx2 + 10, (gy1a + gy2b) // 2
            else:
                do_w = max(wt * 2 + 4, min(fw // 3, fh - wt * 4))
                offset = random.randint(-(fh // 6), fh // 6)
                gy1 = max(sy1 + wt + 2, cy_s + offset - do_w // 2)
                gy2 = min(sy2 - wt - 2, gy1 + do_w);  gy1 = gy2 - do_w
                draw.rectangle([sx2 - wt, gy1, sx2, gy2], fill=(252, 252, 252))
                hx, hy = sx2 - wt, gy1
                draw.line([(hx, hy), (hx - do_w, hy)], fill=(25, 25, 25), width=1)
                draw.arc([hx - do_w, hy - do_w, hx + do_w, hy + do_w],
                         start=90, end=180, fill=(25, 25, 25), width=1)
                do_tx, do_ty = sx2 + 10, (gy1 + gy2) // 2

        lbl_do = random.choice(["D.O.", "DO", "D.O. 900", "DO 800"])
        try:
            draw.text((do_tx, do_ty), lbl_do, fill=gc, font=ann_font, anchor="mm")
        except TypeError:
            draw.text((do_tx - 15, do_ty - 8), lbl_do, fill=gc, font=ann_font)

    # Register the core body so grid columns can't land in its hollow interior.
    col_rects.append((sx1, sy1, sx2, sy2))


# ── CORE-WALL HELPERS ──────────────────────────────────────────────────────────
def _draw_random_unlabeled_core(draw, v_lines, h_lines, gc, ann_font,
                                min_w=70, min_h=50,
                                scale=(0.30, 0.55), text_prob=0.6):
    """Pick a random bay and draw a thick-walled hollow core; return outer bbox."""
    x1, x2, y1, y2 = _random_bay(v_lines, h_lines)
    fw = max(min_w, int((x2 - x1) * random.uniform(*scale)))
    fh = max(min_h, int((y2 - y1) * random.uniform(*scale)))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    sx1, sy1 = cx - fw // 2, cy - fh // 2
    sx2, sy2 = cx + fw // 2, cy + fh // 2
    wt = random.randint(COL_MIN_SIZE // 3, COL_MIN_SIZE // 2)
    _draw_hollow_core(draw, sx1, sy1, sx2, sy2, wt, x_cross=random.random() < 0.5)
    if random.random() < text_prob:
        _text_centered(draw, cx, cy, random.choice(STRUCT_INNER_POOL), gc, ann_font)
    return sx1, sy1, sx2, sy2


# ── CORE WITH LABELED CORNER COLUMNS ──────────────────────────────────────────
def maybe_draw_core_with_corner_columns(draw, v_lines, h_lines, gc, ann_font,
                                        col_rects):
    """Draw a thick-walled hollow core and place labeled columns flush against
    the outside of its corners — teaches detection of columns abutting lift /
    stair / opening walls. Skips a corner if its column would overlap any rect
    in `col_rects` (canvas-wide overlap chokepoint)."""
    if random.random() > CORE_CORNER_PROB or len(v_lines) < 2 or len(h_lines) < 2:
        return []
    sx1, sy1, sx2, sy2 = _draw_random_unlabeled_core(
        draw, v_lines, h_lines, gc, ann_font)
    corners = list(_outer_corners(sx1, sy1, sx2, sy2))
    labels = []
    for (cx_c, cy_c, dx, dy) in random.sample(corners, random.randint(1, 3)):
        # Core-corner distribution: bias toward square+rect (82%) because
        # core/lift/stair corners on real plans are mostly square C-types.
        cls = random.choices([0, 1, 2, 3, 4, 5, 6],
                             weights=[60, 22, 3, 4, 3, 6, 2])[0]
        size = _size_for_cls(cls)
        _, lbl_str = _place_anchored_column(
            draw, cx_c, cy_c, dx, dy, cls, size, col_rects)
        if lbl_str is not None:
            labels.append(lbl_str)
    # Register the hollow-core body so grid columns can't land in its interior.
    col_rects.append((sx1, sy1, sx2, sy2))
    return labels


# ── NEW STRUCTURES: big walled OPENING, 3-WALL STAIR, CHOPPED-WALL LIFT ──────
# Each function draws a real-floor-plan structure on the canvas, places labeled
# true-positive columns at its corners (so the model learns columns DO exist
# adjacent to these structures), and returns the list of YOLO labels for those
# corner columns so the caller can extend its labels list.
#
# True-positive corner-column rate is governed by STRUCT_TP_PROB.

STRUCT_TP_PROB        = 0.90   # per-corner probability of labeled TP column
OPENING_LABEL_INNER   = ["H-D2L2", "H-D2L3", "H-D3L1", "H-D1L2",
                         "H-D2L1", "H-D3L2"]
LIFT_LABEL_INNER      = ["H-D2L1", "300 CIS", "RCB2 800×1000",
                         "LIFT SHAFT", "H-D2L2"]
STAIR_LABEL_INNER     = ["UP", "DN", "DOWN", "UP", "DN"]


def _size_for_cls(cls):
    """Random pixel size for a column of class `cls`, shared by the main-grid
    loop and the core-corner placer (both use the full cls 0..6 scheme):
    rounds (2/4) larger, rects (1/6) biased larger so the short axis stays
    legible, everything else the base square band."""
    if cls in (2, 4):
        return random.randint(ROUND_MIN_SIZE, ROUND_MAX_SIZE)
    if cls in (1, 6):
        return random.randint(COL_MIN_SIZE + 6, COL_MAX_SIZE + 8)
    return random.randint(COL_MIN_SIZE, COL_MAX_SIZE)


def _pick_tp_class_and_size():
    """Pick a (class, size) pair for a TP column placed near a structure.
    Filled square dominates (60%) with rect (20%), filled circle (8%), and
    unshaded round (12%). Cls 4 was previously excluded — real plans DO put
    unshaded round columns at structure corners and along walls, and the
    exclusion produced FNs there. Cls 5/6 (unshaded square/rect) remain out
    of TP roles."""
    cls = random.choices([0, 1, 2, 4], weights=[60, 20, 8, 12])[0]
    if cls in (2, 4):
        return cls, random.randint(ROUND_MIN_SIZE, ROUND_MAX_SIZE)
    if cls == 1:
        return cls, random.randint(COL_MIN_SIZE + 4, COL_MAX_SIZE + 6)
    return cls, random.randint(COL_MIN_SIZE, COL_MAX_SIZE)


def _outer_corners(sx1, sy1, sx2, sy2):
    """Yield (ax, ay, dx, dy) for the 4 outer corners of (sx1,sy1)-(sx2,sy2);
    dx/dy point OUTWARD from the rectangle."""
    yield sx1, sy1, -1, -1
    yield sx2, sy1, +1, -1
    yield sx1, sy2, -1, +1
    yield sx2, sy2, +1, +1


def _outer_edges(sx1, sy1, sx2, sy2):
    """Yield (name, ex1,ey1,ex2,ey2, dx,dy) for the 4 outer edges of
    (sx1,sy1)-(sx2,sy2); dx/dy point OUTWARD from the rectangle."""
    yield "top",    sx1, sy1, sx2, sy1,  0, -1
    yield "bottom", sx1, sy2, sx2, sy2,  0, +1
    yield "left",   sx1, sy1, sx1, sy2, -1,  0
    yield "right",  sx2, sy1, sx2, sy2, +1,  0


def _place_outer_corner_tp_columns(draw, sx1, sy1, sx2, sy2, col_rects):
    """Place labelled TP columns at the 4 outer corners of (sx1,sy1)-(sx2,sy2),
    each gated by `STRUCT_TP_PROB`. Returns the YOLO label strings of every
    placed column."""
    out = []
    for ax, ay, dx, dy in _outer_corners(sx1, sy1, sx2, sy2):
        if random.random() > STRUCT_TP_PROB:
            continue
        _, lbl_str = _place_tp_corner_column(draw, ax, ay, dx, dy, col_rects)
        if lbl_str is not None:
            out.append(lbl_str)
    return out


def _place_anchored_column(draw, ax, ay, dx, dy, cls, size, col_rects):
    """Place a labelled column at anchor (ax, ay) with the body extending
    OUTWARD in direction (dx, dy) ∈ {-1, 0, +1}. The column's INNER face
    sits exactly at the anchor — kissing whatever is at that point (e.g. the
    outer face of a wall slab) without overlapping it.

    Returns (bbox, yolo_label_str) on success; (None, None) if the column
    would overlap any rect already in `col_rects`. On success the padded
    bbox is appended to `col_rects` for the canvas-wide overlap chokepoint.

    Shared by `_place_tp_corner_column` (structure-corner distribution) and
    `maybe_draw_core_with_corner_columns` (core-corner distribution); only
    the (cls, size) pick differs between the two."""
    cx = ax if dx == 0 else ax + dx * (size // 2)
    cy = ay if dy == 0 else ay + dy * (size // 2)
    if _square_overlaps_cols(cx, cy, size, col_rects):
        return None, None
    bbox, _ = place_column(draw, cx, cy, cls, size)
    col_rects.append(_padded_rect(bbox))
    return bbox, _yolo_label(bbox, 0)


def _place_tp_corner_column(draw, ax, ay, dx, dy, col_rects):
    """Place a TP column at corner (ax, ay) with cls/size drawn from the
    structure-corner distribution (`_pick_tp_class_and_size`). See
    `_place_anchored_column` for the placement contract."""
    cls, size = _pick_tp_class_and_size()
    return _place_anchored_column(draw, ax, ay, dx, dy, cls, size, col_rects)


def _place_edge_flanking_columns(draw, x1, y1, x2, y2, dx, dy,
                                 col_rects, gap=4, max_cols=8):
    """Pack labelled TP columns along the axis-aligned segment (x1,y1)→(x2,y2),
    each column's body anchored at the segment with its OUTWARD face extending
    in direction (dx, dy) ∈ {-1, 0, +1}. Adjacent columns are separated by
    `gap` px. A candidate column is skipped (not placed) if its bbox overlaps
    any rectangle already in `col_rects`; this is the no-overlap chokepoint.

    Returns the list of YOLO label strings for newly placed columns. Each
    placed column's bbox is also appended to `col_rects` so subsequent
    flanking calls on other edges of the same structure see it."""
    is_horizontal = (y1 == y2)
    seg_len = (x2 - x1) if is_horizontal else (y2 - y1)
    if seg_len < COL_MIN_SIZE + 2 * gap:
        return []

    labels: list[str] = []
    n_placed = 0
    pos = gap                              # inset from start to avoid stomping corner col
    while n_placed < max_cols:
        cls, size = _pick_tp_class_and_size()
        if pos + size > seg_len - gap:
            break

        if is_horizontal:
            ax = x1 + pos + size // 2
            ay = y1
        else:
            ax = x1
            ay = y1 + pos + size // 2
        cx = ax if dx == 0 else ax + dx * (size // 2)
        cy = ay if dy == 0 else ay + dy * (size // 2)

        if _square_overlaps_cols(cx, cy, size, col_rects):
            pos += size + gap
            continue

        bbox, _ = place_column(draw, cx, cy, cls, size)
        labels.append(_yolo_label(bbox, 0))
        col_rects.append(_padded_rect(bbox))
        n_placed += 1
        pos += size + gap
    return labels


def _place_flanking_with_budget(draw, sx1, sy1, sx2, sy2, col_rects,
                                 budget_lo, budget_hi):
    """Total per-structure budget of `randint(budget_lo, budget_hi)` flanking
    columns sprinkled across the 4 outer edges in random order, with at most
    2 columns per edge. Same shuffled-edges policy that stair / opening /
    lift converged on; centralising it here keeps the three sites in
    lockstep when policy is tweaked."""
    labels = []
    budget = random.randint(budget_lo, budget_hi)
    edges  = list(_outer_edges(sx1, sy1, sx2, sy2))
    random.shuffle(edges)
    for _name, ex1, ey1, ex2, ey2, dx, dy in edges:
        if budget <= 0:
            break
        take   = random.randint(1, min(2, budget))
        placed = _place_edge_flanking_columns(
            draw, ex1, ey1, ex2, ey2, dx, dy, col_rects, max_cols=take)
        labels.extend(placed)
        budget -= len(placed)
    return labels


def _pick_bay_block(v_lines, h_lines, min_bw, max_bw, min_bh, max_bh):
    """Pick a random rectangular block of `bw` × `bh` bays that fits inside
    the grid. Returns (vi, hi, bw, bh) or None if no such block fits."""
    nv, nh = len(v_lines), len(h_lines)
    if nv < min_bw + 1 or nh < min_bh + 1:
        return None
    bw = random.randint(min_bw, min(max_bw, nv - 1))
    bh = random.randint(min_bh, min(max_bh, nh - 1))
    vi = random.randint(0, nv - 1 - bw)
    hi = random.randint(0, nh - 1 - bh)
    return vi, hi, bw, bh


# ── BIG WALLED OPENING (4 thick walls + diagonal X-cross + label) ────────────
# How many candidate bay blocks to try before giving up on placing an opening
# whose footprint is clear. With ~80% grid occupancy and a clean canvas region
# usually existing somewhere, the first 1-2 candidates almost always succeed;
# 30 is a generous floor.
OPENING_PLACEMENT_MAX_TRIES = 30


def _bay_is_clear(bay, col_rects):
    """True if the candidate opening's outer footprint overlaps no rect in
    `col_rects` (existing columns AND already-placed structure footprints).
    This is the contract that keeps the opening's cleared cavity off every
    column: a column is never inside an opening because the opening is never
    placed where a column (or another structure) already is."""
    return not _bbox_overlaps_any(bay, col_rects)


def maybe_draw_opening_big(draw, v_lines, h_lines, col_rects):
    """Draw a multi-bay (2-3 wide × 2-3 tall, biased toward 3) thick-walled
    rectangle with a big diagonal X-cross across the cleared interior and an
    optional "H-D2L2" style label centred. Place labelled true-positive
    columns at the 4 outer corners AND along the 4 outer edges.

    Each candidate bay block is pre-checked with `_bay_is_clear`: if its
    footprint overlaps any existing column or structure rect in `col_rects`,
    the candidate is rejected and another is tried. Openings therefore never
    get drawn on top of a column, so the cleared cavity / X-cross can never
    cover one — the "ghost label inside the X-cross" failure mode flagged on
    the H-D3L2 tile is structurally impossible."""
    if len(v_lines) < 3 or len(h_lines) < 3:
        return []
    labels = []
    n_struct = random.choices([0, 1, 2, 3], weights=[5, 25, 40, 30])[0]
    for _ in range(n_struct):
        for _try in range(OPENING_PLACEMENT_MAX_TRIES):
            # Bias toward 3 bays — real plans have larger void slabs.
            want_bw = 3 if random.random() < 0.70 else 2
            want_bh = 3 if random.random() < 0.70 else 2
            cand = _pick_bay_block(v_lines, h_lines,
                                    min_bw=want_bw, max_bw=want_bw,
                                    min_bh=want_bh, max_bh=want_bh)
            if cand is None:
                continue
            vi, hi, bw, bh = cand
            sx1, sy1 = v_lines[vi],      h_lines[hi]
            sx2, sy2 = v_lines[vi + bw], h_lines[hi + bh]
            if _bay_is_clear((sx1, sy1, sx2, sy2), col_rects):
                break
        else:
            continue   # every candidate bay was occupied — skip this opening

        wt = random.randint(8, 14)
        inner = _draw_hollow_core(draw, sx1, sy1, sx2, sy2, wt, x_cross=False)
        ix1, iy1, ix2, iy2 = inner
        # Big X-cross as a centre-line (chain line) marking the void.
        _draw_x_cross(draw, ix1, iy1, ix2, iy2,
                      color=(20, 20, 20), inset=2, width=2, dashed=True)

        if random.random() < 0.85:
            lbl_sz   = max(28, min(72, (iy2 - iy1) // 4))
            lbl_font = _load_font(lbl_sz)
            cx_m, cy_m = (ix1 + ix2) // 2, (iy1 + iy2) // 2
            lbl = random.choice(OPENING_LABEL_INNER)
            try:
                bb = draw.textbbox((cx_m, cy_m), lbl, font=lbl_font, anchor="mm")
                pad = 6
                draw.rectangle([bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad],
                               fill=(252, 252, 252))
                draw.text((cx_m, cy_m), lbl, fill=(20, 20, 20),
                          font=lbl_font, anchor="mm")
            except TypeError:
                draw.text((cx_m - lbl_sz, cy_m - lbl_sz // 2),
                          lbl, fill=(20, 20, 20), font=lbl_font)

        labels.extend(_place_outer_corner_tp_columns(
            draw, sx1, sy1, sx2, sy2, col_rects))

        # Opening edge-flanking: 0-3 total. 0 floor avoids the "opening
        # always has flanking columns" shortcut — real plans often place
        # an opening in a wall span with no adjacent column.
        labels.extend(_place_flanking_with_budget(
            draw, sx1, sy1, sx2, sy2, col_rects, 0, 3))
        # Register the opening body as occupied so a later grid column whose
        # centre falls inside the cleared cavity is rejected by the overlap
        # gate (the cavity is paper-background and slips past the pixel gate).
        col_rects.append((sx1, sy1, sx2, sy2))
    return labels


# ── 3-WALL STAIR (3 parallel walls + 1 closing wall + steps + zigzag) ────────
def maybe_draw_stair_3wall(draw, v_lines, h_lines, col_rects):
    """Draw a stair shaft with 3 parallel walls (two outer + one middle divider)
    along the long axis, plus a 4th perpendicular wall closing one end. Add
    steps in each flight, a wavy break-line crossing the midpoint of each
    flight, and an "UP" / "DN" label. Flank the OUTSIDE of the 4 corners with
    labeled TP columns."""
    if len(v_lines) < 2 or len(h_lines) < 2:
        return []
    labels = []
    n_struct = random.choices([0, 1, 2, 3], weights=[10, 30, 40, 20])[0]
    for _ in range(n_struct):
        block = _pick_bay_block(v_lines, h_lines,
                                min_bw=1, max_bw=1,
                                min_bh=1, max_bh=2)
        if block is None:
            continue
        vi, hi, bw, bh = block
        # Inset within the bay so the shaft does not touch the grid lines
        pad_x = random.randint(15, 45)
        pad_y = random.randint(15, 45)
        sx1 = v_lines[vi]      + pad_x
        sy1 = h_lines[hi]      + pad_y
        sx2 = v_lines[vi + bw] - pad_x
        sy2 = h_lines[hi + bh] - pad_y
        sw, sh = sx2 - sx1, sy2 - sy1
        if sw < 90 or sh < 120:
            continue

        wt     = random.randint(6, 10)
        half_t = wt // 2
        wall_fill = (random.randint(110, 140),) * 3
        step_col  = (random.randint(60, 90),)  * 3

        # Force long axis along the larger of (sw, sh)
        if sh >= sw:
            mid = sx1 + sw // 2
            # 3 parallel vertical walls
            for x1, x2 in ((sx1,         sx1 + wt),
                           (mid - half_t, mid + half_t),
                           (sx2 - wt,     sx2     )):
                draw.rectangle([x1, sy1, x2, sy2],
                               fill=wall_fill, outline=(10, 10, 10), width=1)
            # 4th wall closing top OR bottom
            if random.random() < 0.5:
                draw.rectangle([sx1, sy1, sx2, sy1 + wt],
                               fill=wall_fill, outline=(10, 10, 10), width=1)
            else:
                draw.rectangle([sx1, sy2 - wt, sx2, sy2],
                               fill=wall_fill, outline=(10, 10, 10), width=1)
            left_a,  left_b  = sx1 + wt,     mid - half_t
            right_a, right_b = mid + half_t, sx2 - wt
            n_steps = max(5, sh // 22)
            step    = sh / n_steps
            for k in range(1, n_steps):
                ty = sy1 + int(k * step)
                draw.line([(left_a,  ty), (left_b,  ty)], fill=step_col, width=1)
                draw.line([(right_a, ty), (right_b, ty)], fill=step_col, width=1)
            # Zigzag break-line across each flight at midpoint
            my = sy1 + sh // 2
            _draw_zigzag(draw, left_a  + 3, my, left_b  - 3, my, step_col)
            _draw_zigzag(draw, right_a + 3, my, right_b - 3, my, step_col)
        else:
            mid = sy1 + sh // 2
            # 3 parallel horizontal walls
            for y1, y2 in ((sy1,         sy1 + wt),
                           (mid - half_t, mid + half_t),
                           (sy2 - wt,     sy2     )):
                draw.rectangle([sx1, y1, sx2, y2],
                               fill=wall_fill, outline=(10, 10, 10), width=1)
            if random.random() < 0.5:
                draw.rectangle([sx1, sy1, sx1 + wt, sy2],
                               fill=wall_fill, outline=(10, 10, 10), width=1)
            else:
                draw.rectangle([sx2 - wt, sy1, sx2, sy2],
                               fill=wall_fill, outline=(10, 10, 10), width=1)
            top_a, top_b = sy1 + wt,     mid - half_t
            bot_a, bot_b = mid + half_t, sy2 - wt
            n_steps = max(5, sw // 22)
            step    = sw / n_steps
            for k in range(1, n_steps):
                tx = sx1 + int(k * step)
                draw.line([(tx, top_a), (tx, top_b)], fill=step_col, width=1)
                draw.line([(tx, bot_a), (tx, bot_b)], fill=step_col, width=1)
            mx = sx1 + sw // 2
            _draw_zigzag(draw, mx, top_a + 3, mx, top_b - 3, step_col)
            _draw_zigzag(draw, mx, bot_a + 3, mx, bot_b - 3, step_col)

        # UP / DN label centered (offset to one side so it doesn't clash with
        # the middle wall divider)
        lbl_sz   = max(22, min(46, min(sw, sh) // 6))
        lbl_font = _load_font(lbl_sz)
        try:
            draw.text((sx1 + sw // 4, sy1 + sh // 2),
                      random.choice(STAIR_LABEL_INNER),
                      fill=(40, 40, 40), font=lbl_font, anchor="mm")
        except TypeError:
            draw.text((sx1 + sw // 4 - lbl_sz, sy1 + sh // 2 - lbl_sz // 2),
                      random.choice(STAIR_LABEL_INNER),
                      fill=(40, 40, 40), font=lbl_font)

        # Bare-stair variant (~30%): no corner columns and no edge flanking,
        # just the stair body. Counter-trains the "stair shape ⇒ column
        # adjacent" prior that produced the marching-strip FPs on real plans
        # — real plans frequently have stairs inside a wall span with no
        # immediately-adjacent labelled columns.
        if random.random() >= 0.30:
            labels.extend(_place_outer_corner_tp_columns(
                draw, sx1, sy1, sx2, sy2, col_rects))
            # Stair edge-flanking: 1-3 total. Real stairs rarely have many
            # flanking columns; the prior dense-strip policy (up to 12) taught
            # the model "stair edge => vertical column row" — the marching FPs
            # we saw on real plans.
            labels.extend(_place_flanking_with_budget(
                draw, sx1, sy1, sx2, sy2, col_rects, 1, 3))
        # Register the shaft body as occupied so grid columns can't land in
        # the open flights between the step lines.
        col_rects.append((sx1, sy1, sx2, sy2))
    return labels


# ── SEGMENTED-WALL LIFT (opening-style box, one wall = door bays) ──────────
def maybe_draw_lift_chopped(draw, v_lines, h_lines, col_rects):
    """Draw a lift shaft: an opening-style grid-aligned walled rectangle with a
    big diagonal X-cross across the cleared cavity and an optional inner label,
    EXCEPT one wall is segmented into several short pieces with gaps between
    them — the lift door bays / human access. Labelled TP columns sit at the 4
    OUTER corners + along the outer edges (kissing the wall, never covered).

    The cross is ALWAYS drawn: a void box only ever lacks a cross when an inner
    column kisses the wall inside it, and this lift places every column OUTSIDE
    the walls. Footprint is pre-checked with `_bay_is_clear` and registered in
    `col_rects` so the cavity never covers a grid column."""
    if len(v_lines) < 2 or len(h_lines) < 2:
        return []
    labels = []
    n_struct = random.choices([0, 1, 2, 3], weights=[5, 30, 40, 25])[0]
    for _ in range(n_struct):
        for _try in range(OPENING_PLACEMENT_MAX_TRIES):
            # Lifts are smaller than openings: 1-2 bays per side.
            want_bw = random.randint(1, 2)
            want_bh = random.randint(1, 2)
            cand = _pick_bay_block(v_lines, h_lines,
                                   min_bw=want_bw, max_bw=want_bw,
                                   min_bh=want_bh, max_bh=want_bh)
            if cand is None:
                continue
            vi, hi, bw, bh = cand
            sx1, sy1 = v_lines[vi],      h_lines[hi]
            sx2, sy2 = v_lines[vi + bw], h_lines[hi + bh]
            if _bay_is_clear((sx1, sy1, sx2, sy2), col_rects):
                break
        else:
            continue   # every candidate bay was occupied — skip this lift
        sw, sh = sx2 - sx1, sy2 - sy1
        if sw < 90 or sh < 90:        # too small for legible door bays + cross
            continue

        wt = random.randint(8, 14)
        ix1, iy1, ix2, iy2 = _draw_hollow_core(
            draw, sx1, sy1, sx2, sy2, wt, x_cross=False)

        # Segment ONE wall into door bays: carve evenly-spaced background gaps,
        # leaving n_gaps+1 short solid wall segments (piers between the doors).
        seg_wall   = random.choice(("top", "bottom", "left", "right"))
        horizontal = seg_wall in ("top", "bottom")
        if horizontal:
            a0, a1   = ix1, ix2
            wy1, wy2 = (sy1, sy1 + wt) if seg_wall == "top" else (sy2 - wt, sy2)
        else:
            a0, a1   = iy1, iy2
            wx1, wx2 = (sx1, sx1 + wt) if seg_wall == "left" else (sx2 - wt, sx2)
        n_gaps = random.randint(2, 4)
        unit   = (a1 - a0) / (2 * n_gaps + 1)   # equal alternating segment/gap
        for k in range(n_gaps):
            g0 = a0 + unit * (2 * k + 1)
            g1 = g0 + unit
            if horizontal:
                draw.rectangle([int(g0), wy1, int(g1), wy2], fill=(252, 252, 252))
            else:
                draw.rectangle([wx1, int(g0), wx2, int(g1)], fill=(252, 252, 252))

        # Big X-cross as a centre-line — ALWAYS (the lift box has no inner column).
        _draw_x_cross(draw, ix1, iy1, ix2, iy2,
                      color=(20, 20, 20), inset=2, width=2, dashed=True)

        # Optional centered label
        if random.random() < 0.85:
            lbl_sz   = max(22, min(48, min(sw, sh) // 7))
            lbl_font = _load_regular_font(lbl_sz)
            cx_m, cy_m = (ix1 + ix2) // 2, (iy1 + iy2) // 2
            lbl = random.choice(LIFT_LABEL_INNER)
            try:
                bb = draw.textbbox((cx_m, cy_m), lbl, font=lbl_font, anchor="mm")
                pad = 4
                draw.rectangle([bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad],
                               fill=(252, 252, 252))
                draw.text((cx_m, cy_m), lbl, fill=(40, 40, 40),
                          font=lbl_font, anchor="mm")
            except TypeError:
                draw.text((cx_m - lbl_sz * 2, cy_m - lbl_sz // 2),
                          lbl, fill=(40, 40, 40), font=lbl_font)

        # Bare-lift variant (~30%): no corner columns and no edge flanking.
        # Same counter-training rationale as bare-stair: gives the model
        # examples of lift bodies that do NOT have any adjacent labelled
        # column, eliminating the "lift edge ⇒ column" prior.
        if random.random() >= 0.30:
            # Labelled TP columns: 4 outer corners + thin outer edge-flanking
            # (2-3 columns total) — the "column next to a segmented wall" signal.
            labels.extend(_place_outer_corner_tp_columns(
                draw, sx1, sy1, sx2, sy2, col_rects))
            labels.extend(_place_flanking_with_budget(
                draw, sx1, sy1, sx2, sy2, col_rects, 2, 3))
        # Register the lift body so grid columns can't land in the cleared cavity.
        col_rects.append((sx1, sy1, sx2, sy2))
    return labels


# ── ZIGZAG BREAK-LINE (used by stairs to mark where a flight is cut) ─────────
def _draw_zigzag(draw, x1, y1, x2, y2, color, amplitude=4, period=10, width=1):
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    tx, ty = (x2 - x1) / length, (y2 - y1) / length
    ax, ay = -ty * amplitude, tx * amplitude
    n_seg = max(2, int(length / period))
    pts = []
    for i in range(n_seg + 1):
        pos = (i / n_seg) * length
        side = -1 if i % 2 == 0 else 1
        pts.append((x1 + tx * pos + ax * side,
                    y1 + ty * pos + ay * side))
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=color, width=width)


# ── EXTRA NEGATIVE CORE WALLS (pure FP suppression) ───────────────────────────
def maybe_draw_negative_core_walls(draw, v_lines, h_lines, gc, ann_font, col_rects):
    """Draw extra unlabeled hollow cores — wall slabs are visually distinct
    from columns (elongated, hollow center), so they're safe negatives."""
    if len(v_lines) < 2 or len(h_lines) < 2:
        return
    for _ in range(2):
        if random.random() < 0.5:
            continue
        rect = _draw_random_unlabeled_core(
            draw, v_lines, h_lines, gc, ann_font,
            min_w=60, min_h=45, scale=(0.25, 0.55), text_prob=0.5)
        # Register the core body so grid columns can't land in its interior.
        col_rects.append(rect)


# ── STANDALONE RC WALL SEGMENTS ────────────────────────────────────────────────
# ── INTERNAL PARTITION WALLS ───────────────────────────────────────────────────
def draw_internal_partitions(img, draw, v_lines, h_lines, col_bboxes,
                              ann_font, gc, col_rects):
    """
    Draw thin (1-2 px) internal partition walls connecting adjacent column faces.

    ~55 % probability per span.  NOT labeled as columns.

    Root-cause fix: in training images all columns are freestanding at grid
    intersections.  In the real 515-column floor plan every column is embedded
    in a wall junction.  Adding these thin lines teaches the model that a dark
    square at the end of a thin wall IS a column, preventing the 30 % miss rate.

    Text labels (WALL_THICK, ROOM_LABEL) are gated against `col_rects` so they
    never overdraw an external column label or another structure rect.
    """
    if len(v_lines) < 2 or len(h_lines) < 2:
        return

    nv     = len(v_lines)
    nh     = len(h_lines)
    ann_sz = max(20, TILE_SIZE // 53)   # matches ann_font_ref in generate_image
    wc     = (random.randint(30, 90),) * 3
    wlw    = random.choice([1, 1, 2])    # mostly 1 px, sometimes 2 px

    def _text_aabb(x, y, lbl, anchor):
        """Approximate AABB of a text drawn at (x,y) with the given anchor.
        Uses ~0.6 × em per character — covers DejaVuSans/Liberation widths for
        the wide ROOM_LABEL strings ('LIFT LOBBY', 'PLANT ROOM') where the
        previous 0.5 × em estimate was 10-15 px short of the rendered glyph."""
        tw = ann_sz * max(1, len(lbl)) * 6 // 10
        th = ann_sz
        if anchor == "mb":     # mid-bottom (text above the anchor)
            return (x - tw // 2, y - th, x + tw // 2, y)
        if anchor == "mm":     # mid-mid
            return (x - tw // 2, y - th // 2, x + tw // 2, y + th // 2)
        if anchor == "v":      # vertical: rotated 90°, _paste_rotated_text
                               # centres the rotated patch at (x, y), so the
                               # AABB is symmetric around it.
            return (x - th // 2, y - tw // 2, x + th // 2, y + tw // 2)
        # tl default
        return (x, y, x + tw, y + th)

    # col_bboxes is ordered: v_lines outer loop, h_lines inner loop
    # Returns None if that intersection was skipped (sparse occupancy).
    def _bb(vi, hi):
        entry = col_bboxes[vi * nh + hi]
        if entry is None:
            return None
        return entry[2]   # (cx, cy, bbox) → bbox [x1,y1,x2,y2]

    # ── Horizontal partition walls ─────────────────────────────────────────────
    # Connect right face of col[vi, hi] to left face of col[vi+1, hi].
    for vi in range(nv - 1):
        for hi in range(nh):
            if random.random() > 0.55:
                continue
            bb_l = _bb(vi,     hi)
            bb_r = _bb(vi + 1, hi)
            if bb_l is None or bb_r is None:
                continue
            x1 = bb_l[2]                                     # right face of left col
            x2 = bb_r[0]                                     # left  face of right col
            cy = (bb_l[1] + bb_l[3] + bb_r[1] + bb_r[3]) // 4   # avg y centre
            if x2 <= x1 + 4:
                continue
            draw.line([(x1, cy), (x2, cy)], fill=wc, width=wlw)
            # Optional wall-thickness label above the wall line
            if random.random() < 0.35:
                lbl   = random.choice(WALL_THICK_POOL)
                mid_x = (x1 + x2) // 2
                if _bbox_overlaps_any(
                        _text_aabb(mid_x, cy - 4, lbl, "mb"), col_rects):
                    continue
                try:
                    draw.text((mid_x, cy - 4), lbl, fill=gc, font=ann_font, anchor="mb")
                except TypeError:
                    draw.text((mid_x - ann_sz, cy - ann_sz - 2), lbl, fill=gc, font=ann_font)

    # ── Vertical partition walls ───────────────────────────────────────────────
    # Connect bottom face of col[vi, hi] to top face of col[vi, hi+1].
    for vi in range(nv):
        for hi in range(nh - 1):
            if random.random() > 0.55:
                continue
            bb_t = _bb(vi, hi)
            bb_b = _bb(vi, hi + 1)
            if bb_t is None or bb_b is None:
                continue
            y1 = bb_t[3]                                     # bottom face of top col
            y2 = bb_b[1]                                     # top    face of bot col
            cx = (bb_t[0] + bb_t[2] + bb_b[0] + bb_b[2]) // 4   # avg x centre
            if y2 <= y1 + 4:
                continue
            draw.line([(cx, y1), (cx, y2)], fill=wc, width=wlw)
            # Optional wall-thickness label (rotated, to the left of the wall)
            if random.random() < 0.35:
                lbl   = random.choice(WALL_THICK_POOL)
                mid_y = (y1 + y2) // 2
                if _bbox_overlaps_any(
                        _text_aabb(cx - 4, mid_y, lbl, "v"), col_rects):
                    continue
                _paste_rotated_text(img, cx - 4, mid_y, lbl, ann_font, gc, angle=90)

    # ── Room name labels in enclosed bay centres ───────────────────────────────
    if random.random() < 0.60:
        for hi in range(nh - 1):
            mid_y  = (h_lines[hi] + h_lines[hi + 1]) // 2
            n_pick = random.randint(0, max(1, nv // 3))
            bays   = random.sample(range(nv - 1), min(n_pick, nv - 1)) if nv > 1 else []
            for vi in bays:
                mid_x = (v_lines[vi] + v_lines[vi + 1]) // 2
                lbl   = random.choice(ROOM_LABEL_POOL)
                if _bbox_overlaps_any(
                        _text_aabb(mid_x, mid_y, lbl, "mm"), col_rects):
                    continue
                try:
                    draw.text((mid_x, mid_y), lbl, fill=gc, font=ann_font, anchor="mm")
                except TypeError:
                    draw.text((mid_x - ann_sz * 2, mid_y - ann_sz // 2),
                              lbl, fill=gc, font=ann_font)


# ── EXTRA INTERIOR AXIS BUBBLES (unlabeled) ───────────────────────────────────
def draw_extra_bubbles(draw, v_lines, h_lines, gc, br, font, bub_lw, col_rects):
    """
    Draw 1–3 extra axis-bubble circles INSIDE the drawing area (not just at
    the margin edge positions).  UNLABELED.

    Addresses FP: in tiling inference a grid bubble can appear centred in a
    tile with no surrounding margin context.  The model needs to see isolated
    circles-with-letters at interior positions and learn they are NOT columns.

    Gated against `col_rects`: a bubble is NEVER drawn over a placed column.
    Up to 10 offset retries per bubble; if all collide the bubble is skipped.
    """
    if random.random() > 0.70:
        return
    n = random.randint(1, 3)
    lbl_choices = list("ABCDEFGHJKLMNP") + [str(i) for i in range(1, 10)]
    for _ in range(n):
        placed = False
        for _try in range(10):
            vi = random.randint(0, len(v_lines) - 1)
            hi = random.randint(0, len(h_lines) - 1)
            ox = random.randint(-br * 3, -br) if random.random() > 0.5 else random.randint(br, br * 3)
            oy = random.randint(-br * 3, -br) if random.random() > 0.5 else random.randint(br, br * 3)
            cx = max(br + 4, min(IMG_WIDTH  - br - 4, v_lines[vi] + ox))
            cy = max(br + 4, min(IMG_HEIGHT - br - 4, h_lines[hi] + oy))
            aabb = (cx - br - 2, cy - br - 2, cx + br + 2, cy + br + 2)
            if not _bbox_overlaps_any(aabb, col_rects):
                placed = True
                break
        if not placed:
            continue
        lbl = random.choice(lbl_choices)
        draw_bubble(draw, cx, cy, lbl, br, gc, font, bub_lw)
        col_rects.append(aabb)


# ── NORTH ARROW / REVISION MARKERS (unlabeled) ─────────────────────────────────
def draw_filled_triangle_markers(draw, col_rects):
    """
    Draw 1–2 filled black triangles (north arrow, revision delta, or section
    marker) at random positions — UNLABELED.

    Addresses FP: the real floor plan has a filled black triangle (north arrow
    symbol / revision cloud marker) at the top of the plan.  The model fired on
    it because it had never seen filled triangles as non-column elements.
    A filled isosceles triangle pointing up ≈ 40–80 px tall is the common form.

    Gated against `col_rects`: a marker is NEVER drawn over a placed column
    (no partial, no complete blocking). Up to 30 placement attempts per
    marker; if all collide, the marker is silently dropped.
    """
    if random.random() > 0.70:
        return
    n = random.randint(1, 2)
    label_font = _load_font(max(28, TILE_SIZE // 45))
    for _ in range(n):
        # Make the north arrow MUCH larger than a column so no cropped portion
        # is column-sized.  h_tri = 186–310 px; base width ~93–155 px.
        h_tri = random.randint(COL_MAX_SIZE * 3, COL_MAX_SIZE * 5)   # 186–310 px
        w_tri = int(h_tri * random.uniform(0.5, 0.8))
        r_circ = int(h_tri * 0.55)
        # Outer envelope = max of compass-circle radius and triangle half-base
        env_w = max(r_circ, w_tri // 2)
        placed = False
        for _try in range(30):
            cx = random.randint(w_tri + 20, IMG_WIDTH  - w_tri - 20)
            cy = random.randint(h_tri + 20, IMG_HEIGHT - h_tri - 20)
            # AABB covers everything: triangle tip at (cy - h_tri), base at
            # (cy + h_tri//4), compass circle from (cy - r_circ) to (cy + r_circ).
            aabb = (cx - env_w, cy - h_tri,
                    cx + env_w, cy + max(r_circ, h_tri // 4))
            if not _bbox_overlaps_any(aabb, col_rects):
                placed = True
                break
        if not placed:
            continue
        fill_v = random.randint(0, 60)
        # Always draw pointing UP (standard north arrow)
        pts = [(cx, cy - h_tri), (cx - w_tri // 2, cy + h_tri // 4),
               (cx, cy), (cx + w_tri // 2, cy + h_tri // 4)]
        draw.polygon(pts, fill=(fill_v, fill_v, fill_v))
        # Compass circle + N label — makes the shape unmistakeable as a north arrow
        draw.ellipse([cx - r_circ, cy - r_circ, cx + r_circ, cy + r_circ],
                     fill=None, outline=(fill_v, fill_v, fill_v), width=2)
        try:
            draw.text((cx, cy - r_circ - 4), "N", fill=(fill_v, fill_v, fill_v),
                      font=label_font, anchor="mb")
        except TypeError:
            draw.text((cx - 10, cy - r_circ - 20), "N",
                      fill=(fill_v, fill_v, fill_v), font=label_font)
        # Register so later drawers also avoid the marker footprint
        col_rects.append(aabb)


# ── SLAB SIGNS ─────────────────────────────────────────────────────────────────
def draw_slab_signs(draw, v_lines, h_lines, gc, ann_font, col_rects):
    """Slab-thickness / drop-panel labels in bay centres — teaches: text ≠ column.
    Gated against `col_rects` so a sign never overdraws a placed column."""
    if random.random() > 0.60 or len(v_lines) < 2 or len(h_lines) < 2:
        return
    ann_sz = max(20, TILE_SIZE // 53)
    for _ in range(random.randint(2, 5)):
        x1, x2, y1, y2 = _random_bay(v_lines, h_lines)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        lbl = random.choice(SLAB_SIGN_POOL)
        tw = ann_sz * max(1, len(lbl)) * 6 // 10
        th = ann_sz
        aabb = (cx - tw // 2, cy - th // 2, cx + tw // 2, cy + th // 2)
        if _bbox_overlaps_any(aabb, col_rects):
            continue
        _text_centered(draw, cx, cy, lbl, gc, ann_font)
        col_rects.append(aabb)


# ── SMALL TEXT DECOYS (column-pixel-sized, including rotated) ─────────────────
def draw_small_text_decoys(img, draw, col_rects, gc):
    """Scatter 10-25 short text tokens at column-pixel sizes (14-28 px) across
    the canvas — UNLABELLED. Closes the FP gap where the real-plan model fired
    on (a) single letters inside beam labels whose glyph silhouette matched a
    small outlined column, and (b) vertical rotated labels like 'RCB3' / 'RS1'
    whose stroke pattern matched a stacked column row.

    Decoys are gated against `col_rects` so they never overlap a real column
    (would create an orphan label). ~40 % of decoys are rotated 90° to cover
    the vertical-text case."""
    if random.random() > 0.85:
        return
    margin = 120
    for _ in range(random.randint(10, 25)):
        sz   = random.randint(14, 28)
        font = _load_regular_font(sz)
        text = random.choice(COLUMN_MIMIC_TOKEN_POOL)
        cx = random.randint(margin, IMG_WIDTH  - margin)
        cy = random.randint(margin, IMG_HEIGHT - margin)
        rotated = random.random() < 0.40
        # Rough AABB — short tokens at small sizes don't need exact font metrics
        if rotated:
            cand_w, cand_h = sz + 2, sz * len(text)
        else:
            cand_w, cand_h = sz * max(1, len(text)) // 2 + sz, sz
        cand = (cx - cand_w // 2, cy - cand_h // 2,
                cx + cand_w // 2, cy + cand_h // 2)
        if _bbox_overlaps_any(cand, col_rects):
            continue
        if rotated:
            _paste_rotated_text(img, cx, cy, text, font, gc, angle=90)
        else:
            _text_centered(draw, cx, cy, text, gc, font)


# ── EXTERNAL COLUMN LABELS (adjacent to ~40 % of grid columns) ────────────────
def _label_placement(bbox, side, sz, token, angle, gap=6):
    """Return (cx, cy, cand_aabb) for an external column label placed on the
    given `side` of `bbox` with a small `gap`. `cand_aabb` is a conservative
    AABB used to gate against `col_rects` — overestimated for 45° rotations
    (we use a square envelope of `max(tw, sz)` instead of solving the exact
    rotated rect)."""
    x1, y1, x2, y2 = bbox
    tw = sz * max(1, len(token))
    if side == "top":
        cx = (x1 + x2) // 2
        cy = y1 - gap - sz // 2
    elif side == "bottom":
        cx = (x1 + x2) // 2
        cy = y2 + gap + sz // 2
    elif side == "left":
        cx = x1 - gap - tw // 2
        cy = (y1 + y2) // 2
    else:  # right
        cx = x2 + gap + tw // 2
        cy = (y1 + y2) // 2
    if angle == 0:
        cand_w, cand_h = tw, sz
    elif angle == 90:
        cand_w, cand_h = sz, tw
    else:  # 45 — diagonal envelope
        side_env = max(tw, sz)
        cand_w = cand_h = side_env
    cand = (cx - cand_w // 2, cy - cand_h // 2,
            cx + cand_w // 2, cy + cand_h // 2)
    return cx, cy, cand


def draw_column_labels(img, draw, col_bboxes, col_rects):
    """Place an external column label adjacent to ~40 % of grid columns.
    Rotation is uniformly 0° / 45° / 90° to mirror the real-plan convention
    where labels sit flat, slanted, or stacked vertically next to the
    column. The label is NOT bbox'd — text is decorative; the column itself
    already has its YOLO label. Gated against `col_rects` so the text never
    overlays another column or structure body. Replaces the prior cls 4
    interior-text approach which trained the model on a layout that does
    not occur in production drawings."""
    for entry in col_bboxes:
        if entry is None:
            continue
        _cx, _cy, bbox = entry
        if random.random() > 0.40:
            continue
        raw   = random.choice(COLUMN_LABEL_POOL)
        # Always pick the FIRST line (the column code, e.g. "C2") — never
        # the size suffix alone. Real plans label a column with its code,
        # never with a bare dimension string ("800×800") which would teach
        # the model that any small adjacent dimension fragment is a column
        # tag.  Single-line ensures the AABB math (cand_h = sz, cand_w =
        # sz × len(token)) matches the rendered text.
        token = raw.split("\n")[0]
        sz    = random.randint(14, 22)
        font  = _load_regular_font(sz)
        angle = random.choice([0, 45, 90])
        # Try each side; first non-overlapping placement wins. If all four
        # collide, this column gets no label this canvas — fine.
        for side in random.sample(("top", "bottom", "left", "right"), 4):
            tx, ty, cand = _label_placement(bbox, side, sz, token, angle)
            # Skip placements that would push _paste_rotated_text to clamp
            # the paste origin inward — that would silently shift the
            # rendered text off the registered cand AABB.
            if (cand[0] < 0 or cand[1] < 0
                    or cand[2] >= IMG_WIDTH or cand[3] >= IMG_HEIGHT):
                continue
            if _bbox_overlaps_any(cand, col_rects):
                continue
            if angle == 0:
                _text_centered(draw, tx, ty, token, (40, 40, 40), font)
            else:
                _paste_rotated_text(img, tx, ty, token, font,
                                     (40, 40, 40), angle=angle)
            # Register the label's footprint so partition lines / later
            # drawers don't bisect it.
            col_rects.append(cand)
            break


# ── EMPTY-INTERSECTION DECOYS ──────────────────────────────────────────────────
def draw_empty_intersection_decoys(draw, v_lines, h_lines, placed_xy, gc, ann_font):
    """Column/beam labels at empty intersections — breaks the
    'C2-text ⇒ column nearby' shortcut the model learns from biased layouts."""
    if random.random() > 0.85 or not v_lines or not h_lines:
        return
    candidates = [
        (x, y) for x in v_lines for y in h_lines
        if not any(abs(x - px) < 8 and abs(y - py) < 8 for (px, py) in placed_xy)
    ]
    if not candidates:
        return
    n = min(len(candidates), random.randint(6, 18))
    for (x, y) in random.sample(candidates, n):
        if random.random() < 0.5:
            # Take only the code line ("C2") — never the bare size suffix
            # ("800×800") alone. Matches draw_column_labels' policy.
            lbl = random.choice(COLUMN_LABEL_POOL).split("\n")[0]
        else:
            lbl = random.choice(BEAM_LABEL_H_POOL)
        # Place as if it were captioning a column: small offset
        tx = x + random.choice([-1, 1]) * random.randint(6, 14)
        ty = y - random.randint(0, 14)
        try:
            draw.text((tx, ty), lbl, fill=gc, font=ann_font)
        except Exception:
            pass


# ── GRID-CROSSING DECOYS (solid-line T/+ junctions) ────────────────────────────
def draw_grid_crossing_decoys(draw, v_lines, h_lines, col_rects):
    """Short solid-line + and T crossings scattered between grid intersections.
    Real plans have wall-edge / beam-edge crossings the model mistakes for tiny
    columns; explicit unlabeled examples teach suppression. Gated against
    `col_rects` so a decoy never overlaps a placed column."""
    if random.random() > 0.70 or len(v_lines) < 2 or len(h_lines) < 2:
        return
    for _ in range(random.randint(8, 20)):
        x1, x2, y1, y2 = _random_bay(v_lines, h_lines)
        if x2 - x1 < COL_MIN_SIZE * 4 or y2 - y1 < COL_MIN_SIZE * 4:
            continue
        cx = random.randint(x1 + 20, x2 - 20)
        cy = random.randint(y1 + 20, y2 - 20)
        arm = random.randint(COL_MIN_SIZE, COL_MAX_SIZE + 8)
        aabb = (cx - arm, cy - arm, cx + arm, cy + arm)
        if _bbox_overlaps_any(aabb, col_rects):
            continue
        col = (random.randint(30, 90),) * 3
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=col, width=1)
        if random.random() < 0.6:
            draw.line([(cx, cy - arm), (cx, cy + arm)], fill=col, width=1)
        else:
            draw.line([(cx, cy), (cx, cy + arm)], fill=col, width=1)
        col_rects.append(aabb)


# ── OUTER BORDER RECTANGLE ─────────────────────────────────────────────────────
def maybe_draw_border(draw, w, h, margin):
    """
    Draw a thin outer border rectangle in ~65 % of images.
    The real floor plan has a red border; training needs many examples of a
    large rectangle that is NOT a column.
    """
    if random.random() > 0.65:
        return
    bm  = margin // 2
    lw  = random.randint(1, 3)
    vc  = random.randint(80, 180)
    draw.rectangle([bm, bm, w - bm, h - bm],
                   fill=None, outline=(vc, vc, vc), width=lw)


# ── SPLIT ASSIGNMENT ──────────────────────────────────────────────────────────
def get_split(i, n):
    p = i / n
    if p < TRAIN_RATIO:
        return "train"
    elif p < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"


# ── TILE SAVER ─────────────────────────────────────────────────────────────────
def _save_tiles(img, labels, split, image_id):
    """
    Cut the large canvas into TILE_SIZE×TILE_SIZE patches (step = TILE_STEP)
    and save each patch + its adjusted YOLO labels.

    Column labels whose centre falls within a tile are kept; their normalised
    coordinates are adjusted to tile-local space.  Tiles with no columns are
    saved with empty label files (negative examples — important for FP suppression).

    NOTE: in the 200-px overlap zones a column's centre may fall in 2-4
    tiles. The column IS labelled in every such tile (intentional duplicate
    — standard YOLO tiled-training practice). Inference-time dedup is the
    job of `scripts/postprocess_detections.py` (cross-tile NMS). Labelling
    in only one tile would leave the column VISIBLE but unlabelled in the
    adjacent tile, training the model to suppress its own positives.
    """
    W, H = img.width, img.height

    # Build column list in pixel coordinates (from normalised canvas coords).
    # `_label_to_pixels` is the canonical YOLO-string parser and is valid here
    # because the canvas is always IMG_WIDTH × IMG_HEIGHT.
    col_pixels = [_label_to_pixels(lbl) for lbl in labels]

    # Tile positions (include a final tile flush against the right/bottom edge)
    xs = list(range(0, W - TILE_SIZE, TILE_STEP))
    if not xs or xs[-1] + TILE_SIZE < W:
        xs.append(max(0, W - TILE_SIZE))
    ys = list(range(0, H - TILE_SIZE, TILE_STEP))
    if not ys or ys[-1] + TILE_SIZE < H:
        ys.append(max(0, H - TILE_SIZE))

    tile_idx = 0
    pos_tiles = 0
    neg_tiles = 0
    for tx in xs:
        for ty in ys:
            tx2, ty2 = tx + TILE_SIZE, ty + TILE_SIZE
            tile = img.crop((tx, ty, tx2, ty2))

            tile_labels = []
            tile_rects  = []   # pixel-space rects, reused by the human_check overlay
            for cls_id, bx_px, by_px, bw_px, bh_px in col_pixels:
                # Include only columns whose centre is inside this tile
                if not (tx <= bx_px < tx2 and ty <= by_px < ty2):
                    continue
                # Adjust centre to tile-local normalised coords
                cx_t = (bx_px - tx) / TILE_SIZE
                cy_t = (by_px - ty) / TILE_SIZE
                bw_t = min(bw_px, TILE_SIZE) / TILE_SIZE
                bh_t = min(bh_px, TILE_SIZE) / TILE_SIZE
                cx_t = max(0.001, min(0.999, cx_t))
                cy_t = max(0.001, min(0.999, cy_t))
                bw_t = max(0.005, min(0.999, bw_t))
                bh_t = max(0.005, min(0.999, bh_t))
                tile_labels.append(
                    f"{cls_id} {cx_t:.6f} {cy_t:.6f} {bw_t:.6f} {bh_t:.6f}"
                )
                half_w = bw_t * TILE_SIZE / 2
                half_h = bh_t * TILE_SIZE / 2
                cx_px  = cx_t * TILE_SIZE
                cy_px  = cy_t * TILE_SIZE
                tile_rects.append((int(cx_px - half_w), int(cy_px - half_h),
                                   int(cx_px + half_w), int(cy_px + half_h)))

            # File name: image_id × 10000 + tile_idx (supports up to 9999 tiles/image)
            tid = image_id * 10000 + tile_idx
            fname = f"{tid:08d}"
            tile.save(os.path.join(OUTPUT_DIR, "images", split, f"{fname}.png"))
            with open(os.path.join(OUTPUT_DIR, "labels", split, f"{fname}.txt"), "w") as f:
                f.write("\n".join(tile_labels) + ("\n" if tile_labels else ""))

            # Reuse `tile` (clean copy already on disk) — skip .copy() to save
            # ~5 MB/tile. JPG q=85 is plenty for QA and is several × faster
            # than PNG for the overlay save.
            if HUMAN_CHECK and tile_rects:
                overlay = ImageDraw.Draw(tile)
                for r in tile_rects:
                    overlay.rectangle(r, outline=(255, 0, 0), width=2)
                tile.save(os.path.join(OUTPUT_DIR, "human_check",
                                       split, f"{fname}.jpg"),
                          "JPEG", quality=85)

            tile_idx += 1
            if tile_labels:
                pos_tiles += 1
            else:
                neg_tiles += 1

    return tile_idx, pos_tiles, neg_tiles


# ── IMAGE GENERATION ───────────────────────────────────────────────────────────
def generate_image(image_id, split, negative=False):
    bg   = random.randint(248, 255)
    img  = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (bg, bg, bg))
    draw = ImageDraw.Draw(img)

    labels = []
    placed = []

    v_lines, h_lines = make_grid(IMG_WIDTH, IMG_HEIGHT)

    # ── Grid line style ────────────────────────────────────────────────────────
    dash = random.randint(12, 28)
    gap  = random.randint(8,  18)
    gv   = random.randint(50, 90)
    gc   = (gv, gv, gv)
    glw  = random.randint(1, 2)

    # ── Bubble parameters ──────────────────────────────────────────────────────
    # Radius calibrated for 2560-px canvas: 59-69 px → 29-35 px at inference.
    # Real floor plan bubbles appear as ~30-35 px at inference.  ✓
    br      = random.randint(TILE_SIZE // 30, TILE_SIZE // 26)   # 59–69 px
    font_sz = max(20, int(br * 0.92))
    font    = _load_font(font_sz)
    bub_lw  = max(2, br // 20)
    margin  = 12

    top_by  = br + margin
    bot_by  = IMG_HEIGHT - br - margin
    lft_bx  = br + margin
    rgt_bx  = IMG_WIDTH  - br - margin

    # ── Outer border rectangle ─────────────────────────────────────────────────
    maybe_draw_border(draw, IMG_WIDTH, IMG_HEIGHT, margin)

    # ── Dashed grid lines ──────────────────────────────────────────────────────
    for x in v_lines:
        dashed_line(draw, x, top_by, x, bot_by, dash=dash, gap=gap, color=gc, w=glw)
    for y in h_lines:
        dashed_line(draw, lft_bx, y, rgt_bx, y, dash=dash, gap=gap, color=gc, w=glw)

    # ── Axis bubbles (A, B, C … / 1, 2, 3 …) ─────────────────────────────────
    # 30 % of bubbles get a sub-label below them (e.g. "300-BR-01", "EXP-JT")
    # to match the real floor plan where bubble sub-labels caused FP detections.
    sub_label_pool = ["300-BR-01", "EXP-JT", "GL-01", "EXP", "REF", "300-CL"]
    sub_sz   = max(14, TILE_SIZE // 85)
    sub_font = _load_regular_font(sub_sz)
    if random.random() > 0.08:
        for i, x in enumerate(v_lines):
            lbl = chr(65 + i % 26)
            draw_bubble(draw, x, top_by, lbl, br, gc, font, bub_lw)
            draw_bubble(draw, x, bot_by, lbl, br, gc, font, bub_lw)
            if random.random() < 0.30:
                try:
                    draw.text((x, top_by + br + 2), random.choice(sub_label_pool),
                              fill=gc, font=sub_font, anchor="mt")
                except TypeError:
                    draw.text((x - sub_sz * 3, top_by + br + 2),
                              random.choice(sub_label_pool), fill=gc, font=sub_font)
        for i, y in enumerate(h_lines):
            lbl = str(i + 1)
            draw_bubble(draw, lft_bx, y, lbl, br, gc, font, bub_lw)
            draw_bubble(draw, rgt_bx, y, lbl, br, gc, font, bub_lw)
            if random.random() < 0.30:
                try:
                    draw.text((lft_bx, y + br + 2), random.choice(sub_label_pool),
                              fill=gc, font=sub_font, anchor="mt")
                except TypeError:
                    draw.text((lft_bx - sub_sz * 3, y + br + 2),
                              random.choice(sub_label_pool), fill=gc, font=sub_font)

    ann_sz_ref = max(20, TILE_SIZE // 53)   # annotation reference size
    ann_font_ref = _load_regular_font(ann_sz_ref)

    # Canvas-wide occupancy list — padded column bboxes (structure-edge
    # AND main-grid columns) AND structure footprints (opening / stair /
    # lift / core bodies). The no-overlap chokepoint shared by structure
    # corner / edge-flanking placements, the grid-column gate, and any late
    # drawer that gates on col_rects (currently draw_small_text_decoys), so
    # nothing is drawn over a column or inside a cleared structure cavity.
    # Structure footprints are stored UNPADDED (raw outer wall extent) —
    # unlike column rects — so a TP column may legitimately kiss a
    # structure's outer wall; the gate pads the candidate instead.
    col_rects: list = []

    # ── STRUCTURES FIRST ───────────────────────────────────────────────────────
    # Walls / cores / openings / stairs / lifts are drawn BEFORE the grid
    # columns so that grid columns can be placed last and skip any spot a
    # structure already occupies. This is what guarantees a column is never
    # drawn over or under a wall — the failure mode in the QA screenshots.
    if random.random() < 0.30:
        maybe_draw_special_structure(draw, v_lines, h_lines, gc, ann_font_ref, col_rects)
    maybe_draw_negative_core_walls(draw, v_lines, h_lines, gc, ann_font_ref, col_rects)
    if not negative:
        labels.extend(maybe_draw_core_with_corner_columns(
            draw, v_lines, h_lines, gc, ann_font_ref, col_rects))
        labels.extend(maybe_draw_opening_big(
            draw, v_lines, h_lines, col_rects))
        labels.extend(maybe_draw_stair_3wall(
            draw, v_lines, h_lines, col_rects))
        labels.extend(maybe_draw_lift_chopped(
            draw, v_lines, h_lines, col_rects))

    # ── GRID COLUMNS LAST ──────────────────────────────────────────────────────
    # Sparse occupancy: 20% of intersections have NO column (real plans have
    # empty intersections — trains FP suppression). Each candidate is gated so
    # it never overlaps a structure column nor lands on a wall / structure.
    pixels = img.load()
    # Snapshot of the structure columns only — grid intersections are 190-310 px
    # apart so grid-vs-grid overlap is geometrically impossible; checking only
    # the fixed structure list keeps the gate O(structures) instead of O(N²).
    struct_rects = list(col_rects)
    col_bboxes = []
    for x in v_lines:
        for y in h_lines:
            # Negative canvases skip every intersection so the resulting tiles
            # contain only background grid / bubbles with empty labels.
            if negative or random.random() < 0.20:
                col_bboxes.append(None)          # placeholder to keep grid index intact
                continue
            # Main-grid distribution: full cls 0..6 coverage. Square+rect
            # dominate (~75%); the smaller round/combined/unshaded slices
            # ensure the model sees every variant (incl. outline-only squares).
            # cls 4 (unshaded round) bumped 6 → 15 to lift the model's hit
            # rate on real-plan unshaded round columns; the absolute count
            # was previously ~5 per canvas and the FN floor was high.
            cls  = random.choices([0, 1, 2, 3, 4, 5, 6],
                                  weights=[55, 28, 6, 8, 15, 7, 3])[0]
            size = _size_for_cls(cls)
            if cls in (0, 5) and random.random() < SMALL_COL_PROB:
                size = random.randint(SMALL_COL_MIN_SIZE, SMALL_COL_MAX_SIZE)
            jx, jy = random.randint(-3, 3), random.randint(-3, 3)
            cx, cy = x + jx, y + jy
            # Gate: skip if the padded label would overlap a structure column,
            # or if the footprint pixels are already occupied by a wall /
            # structure. Either way the column is never drawn over another element.
            if (_square_overlaps_cols(cx, cy, size, struct_rects)
                    or not _spot_is_clear(pixels, cx, cy, size)):
                col_bboxes.append(None)
                continue
            bbox, yolo_cls = place_column(draw, cx, cy, cls, size)
            placed.append((cx, cy))
            col_bboxes.append((cx, cy, bbox))
            # Expose grid columns to late drawers that gate on col_rects
            # (currently draw_small_text_decoys). The struct_rects snapshot
            # was already taken at :1955, so the grid loop's own gate is
            # unaffected — this list only grows for downstream consumers.
            col_rects.append(_padded_rect(bbox))
            labels.append(_yolo_label(bbox, yolo_cls))

    # ── External column labels (text adjacent to ~40 % of grid columns) ────────
    # Real plans tag columns OUTSIDE the shape (flat / slanted / vertical),
    # not inside. Drawn before partitions so partition lines avoid the new
    # label rectangles via col_rects.
    draw_column_labels(img, draw, col_bboxes, col_rects)

    # ── Internal partition walls (thin lines connecting column faces) ──────────
    draw_internal_partitions(img, draw, v_lines, h_lines, col_bboxes,
                              ann_font_ref, gc, col_rects)

    # ── Extra interior axis bubbles (unlabeled circles near grid, not columns) ─
    draw_extra_bubbles(draw, v_lines, h_lines, gc, br, font, bub_lw, col_rects)

    # ── Filled triangle markers / north arrows (unlabeled) ────────────────────
    draw_filled_triangle_markers(draw, col_rects)

    # ── Slab signs / drop-panel labels in bays (unlabeled) ────────────────────
    draw_slab_signs(draw, v_lines, h_lines, gc, ann_font_ref, col_rects)

    # Column-pixel-sized text decoys (single letters, short tokens, rotated).
    # Gated against col_rects so they never overlap a real column.
    draw_small_text_decoys(img, draw, col_rects, gc)

    draw_empty_intersection_decoys(draw, v_lines, h_lines, placed, gc, ann_font_ref)
    draw_grid_crossing_decoys(draw, v_lines, h_lines, col_rects)

    pixels = img.load()
    labels = [lbl for lbl in labels if not _is_orphan_label(lbl, pixels)]

    # ── Optional debug overlay (must run AFTER the orphan scrub so dropped
    #    labels aren't drawn over) ─────────────────────────────────────────
    if DRAW_DEBUG_BOXES:
        dbg_font = _load_font(max(14, TILE_SIZE // 90))
        for line in labels:
            _, cx, cy, bw, bh = _label_to_pixels(line)
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=3)
            draw.text((x1 + 2, y1 - ann_sz_ref - 2), "col", fill=(220, 30, 30), font=dbg_font)

    # ── Tile the large canvas and save each tile ──────────────────────────────
    # Labels are in normalised canvas coordinates; _save_tiles() adjusts them
    # to each tile's coordinate system.  This is exactly what inference does.
    return _save_tiles(img, labels, split, image_id)


# ── CANVAS PIPELINE ENTRY POINT ───────────────────────────────────────────────
def run_canvas_pipeline(num_canvases: int = NUM_IMAGES,
                        negative_ratio: float = NEGATIVE_RATIO,
                        start_index: int = START_INDEX) -> tuple[int, int]:
    """Run the A0-canvas → tile pipeline. Returns (positive_tiles, negative_tiles)."""
    splits = [get_split(i, num_canvases) for i in range(num_canvases)]
    counts = {s: splits.count(s) for s in ("train", "val", "test")}

    n_neg   = int(round(num_canvases * negative_ratio))
    neg_idx = set(random.sample(range(num_canvases), n_neg))

    nx = len(list(range(0, IMG_WIDTH  - TILE_SIZE, TILE_STEP))) + 1
    ny = len(list(range(0, IMG_HEIGHT - TILE_SIZE, TILE_STEP))) + 1
    tiles_per = nx * ny
    print(f"[canvas] {num_canvases} canvases ({IMG_WIDTH}×{IMG_HEIGHT} px)")
    print(f"[canvas]   tile {TILE_SIZE}×{TILE_SIZE} step {TILE_STEP} → {nx}×{ny} = {tiles_per}/canvas")
    print(f"[canvas]   ~{num_canvases * tiles_per} tiles · "
          f"train:{counts['train']} val:{counts['val']} test:{counts['test']} canvases")
    print(f"[canvas]   negatives {n_neg}/{num_canvases} ({negative_ratio:.0%})")

    total_pos, total_neg = 0, 0
    for i in range(num_canvases):
        is_neg = i in neg_idx
        _, pos, neg = generate_image(start_index + i, splits[i], negative=is_neg)
        total_pos += pos
        total_neg += neg
        tag = "NEG" if is_neg else "pos"
        print(f"[canvas]   {i + 1:>3}/{num_canvases} ({splits[i]} {tag}) → "
              f"{pos} pos + {neg} neg tiles")
    print(f"[canvas] done — {total_pos + total_neg} tiles "
          f"({total_pos} pos · {total_neg} neg)")
    return total_pos, total_neg




# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def _main(argv=None) -> int:
    global OUTPUT_DIR, HUMAN_CHECK
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out",            default=OUTPUT_DIR,
                        help=f"Output dataset root (default: {OUTPUT_DIR})")
    parser.add_argument("--canvases",       type=int,   default=NUM_IMAGES,
                        help=f"Number of A0 canvases to generate (default: {NUM_IMAGES})")
    parser.add_argument("--negative-ratio", type=float, default=NEGATIVE_RATIO,
                        help=f"Fraction of canvases with no columns (default: {NEGATIVE_RATIO})")
    parser.add_argument("--start-index",    type=int,   default=START_INDEX,
                        help=f"First canvas id, for resuming (default: {START_INDEX})")
    parser.add_argument("--clean",          action="store_true",
                        help="Delete the output dataset directory before generating")
    parser.add_argument("--no-human-check", action="store_true",
                        help="Skip writing annotated overlay tiles to "
                             "dataset/column/human_check/. Each overlay is a "
                             "JPG re-encode of the labelled tile; with ~100 "
                             "labelled tiles per canvas × 200 canvases that "
                             "is the single biggest wall-clock cost (~10 min "
                             "on a 200-canvas run). Pass this flag for bulk "
                             "regens once labels have been QA'd.")
    args = parser.parse_args(argv)

    OUTPUT_DIR  = args.out
    HUMAN_CHECK = not args.no_human_check

    if args.clean and os.path.isdir(OUTPUT_DIR):
        print(f"Removing existing dataset at {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    create_dirs()
    write_dataset_meta()

    print(f"Output      : {OUTPUT_DIR}/")
    print(f"Human-check : {'on' if HUMAN_CHECK else 'off'}"
          + (f"  → {OUTPUT_DIR}/human_check/" if HUMAN_CHECK else ""))
    print(f"Tile size   : {TILE_SIZE}×{TILE_SIZE} px")
    print()

    pos, neg = run_canvas_pipeline(
        num_canvases   = args.canvases,
        negative_ratio = args.negative_ratio,
        start_index    = args.start_index,
    )

    print()
    print(f"Dataset → {OUTPUT_DIR}/")
    print(f"  tiles     : {pos + neg}  ({pos} positive · {neg} negative)")
    print(f"  data.yaml : {OUTPUT_DIR}/data.yaml")
    if HUMAN_CHECK:
        print(f"  overlays  : {OUTPUT_DIR}/human_check/{{train,val,test}}/")
    print()
    print("Train with: python3 train.py")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
