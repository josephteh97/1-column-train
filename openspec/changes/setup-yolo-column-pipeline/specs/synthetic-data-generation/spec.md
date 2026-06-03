## ADDED Requirements

### Requirement: Tile geometry invariant

The synthetic generator SHALL emit training tiles at `TILE_SIZE = 1280` pixels
square with `TILE_STEP = 1080` pixels (200-pixel overlap). All column-sizing
constants (e.g. `COL_MIN_SIZE`, `COL_MAX_SIZE`, `ROUND_MIN_SIZE`,
`ROUND_MAX_SIZE`) MUST be calibrated to the post-tile pixel scale, so a
column rendered at training time has the same per-tile pixel size it would
have at inference time on a real plan tiled with the same geometry.

#### Scenario: Tile size used at training matches inference
- **WHEN** the training pipeline reads tiles and the inference pipeline tiles a real plan
- **THEN** both use `TILE_SIZE = 1280` and `TILE_STEP = 1080`, and the YOLO `imgsz` argument matches `TILE_SIZE`

#### Scenario: Changing tile geometry without recalibrating column sizes is rejected
- **WHEN** a developer changes `TILE_SIZE` or `TILE_STEP` in `generate_column.py` without changing the column-size constants
- **THEN** code review SHALL flag the change as a baseline-breaking regression

### Requirement: No column blocking

Late drawers MUST NOT paint ink over any rectangle in `col_rects`. Every drawer in `generate_image` that runs after the column-placement phase SHALL gate every drawn element against `col_rects` and skip placements that would overlap.

#### Scenario: Late drawer attempts to paint over a column
- **WHEN** a drawer such as `draw_filled_triangle_markers`, `draw_extra_bubbles`, `draw_slab_signs`, or `draw_grid_crossing_decoys` computes a candidate AABB
- **THEN** it MUST check `_bbox_overlaps_any(aabb, col_rects)` before drawing, and skip (or retry) on overlap

#### Scenario: New drawer is added later
- **WHEN** a contributor adds a new late drawer that paints ink
- **THEN** the drawer MUST accept `col_rects` as a parameter, gate every element against it, and append the drawn AABB to `col_rects` so subsequent drawers also avoid it

### Requirement: Per-tile centre-in-tile labelling without dedup

The tile saver SHALL emit a column label in EVERY tile whose extent contains
the column's centre. Duplicate labels in the 200-px overlap zones are
required behaviour; cross-tile dedup MUST NOT be performed at training time.

#### Scenario: Column centre falls in overlap zone
- **WHEN** a column's centre lies within the overlap region of two adjacent tiles
- **THEN** the column SHALL be labelled in BOTH tiles, with normalised coordinates in tile-local space

#### Scenario: Same column appears in adjacent overlap tile
- **WHEN** a column is visible in two tiles
- **THEN** the tile saver MUST NOT pick a single owning tile and leave the other tile showing the column unlabelled

### Requirement: Tight YOLO label bbox with 1-px margin

Saved YOLO label bboxes SHALL inscribe the actual drawn column extent with a
fixed `LABEL_PAD = 1` pixel margin on each side. The same constant SHALL be
consumed by `_yolo_label`, `_padded_rect` (for placement spacing), and
`_is_orphan_label` (for the sample-position formula).

#### Scenario: Default label production
- **WHEN** `place_column` returns a shape bbox
- **THEN** `_yolo_label(bbox, cls)` produces a YOLO label with width `actual_width + 2` and height `actual_height + 2` (1-px margin per side)

#### Scenario: Label-pad consistency
- **WHEN** any of `_yolo_label`, `_padded_rect`, or `_is_orphan_label` reference label padding
- **THEN** they SHALL use the single module-level `LABEL_PAD` constant; no per-function override

### Requirement: Single-class YOLO output

Saved YOLO labels MUST use class id `0` for every column variant. The internal generator `cls` ids 0-6 SHALL control rendered geometry only and MUST be erased before the label is written to disk.

#### Scenario: Geometry variant emitted
- **WHEN** `place_column` is called with any `cls` value in `0..6`
- **THEN** the returned YOLO label SHALL have class `0`

### Requirement: Bare-stair / bare-lift variants

Approximately 30% of stair instances and 30% of lift instances SHALL be
rendered WITHOUT corner T/+ columns AND WITHOUT edge flanking columns.
This counter-trains the "stair/lift edge ⇒ adjacent column" prior.

#### Scenario: Bare-stair roll
- **WHEN** the per-stair random roll falls in the bare-stair branch
- **THEN** neither `_place_outer_corner_tp_columns` nor `_place_flanking_with_budget` is called for that stair, but all other stair drawing (walls, treads, zigzag, UP/DN label) proceeds normally

#### Scenario: Empirical rate is preserved
- **WHEN** the generator is run over a large batch (200+ stairs)
- **THEN** approximately 30% (±5%) of stairs are bare, verifiable by hooking the placer functions

### Requirement: Orphan-label scrub as safety net

The orphan scrub `_is_orphan_label` SHALL remove any label whose 5 sample
pixels (centre + 4 outline mid-points at `bw/2 - LABEL_PAD - 1`) are all
paper-background. It is the final safety net before the label file is
written.

#### Scenario: Drawer overpaints a column
- **WHEN** a late drawer or structure cavity inadvertently fills the column body with background
- **THEN** the orphan scrub SHALL detect and remove the now-empty label

### Requirement: Human-check overlay generation

For every positive tile saved with non-empty labels, the generator SHALL
also emit a JPG into `human_check/{split}/` with each labelled column
outlined in red. This is the manual QA channel.

#### Scenario: Tile with at least one label
- **WHEN** `_save_tiles` writes a positive tile
- **THEN** a parallel JPG with red bbox overlays SHALL be written to `human_check/<split>/`, named identically except for the `.jpg` extension
