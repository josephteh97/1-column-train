## ADDED Requirements

### Requirement: Real plan ingestion at configurable DPI

The data pipeline SHALL ingest real annotated floor-plan PDFs and
images at a single configurable input DPI, rasterising PDFs to PNG /
JPG at that DPI before tiling. The DPI value used for ingestion MUST
be recorded alongside the raw image so downstream training and
inference can verify the calibration.

#### Scenario: PDF ingestion at the configured DPI
- **WHEN** a real-plan PDF is ingested with `INPUT_DPI=300`
- **THEN** the page is rasterised at 300 DPI, the resulting image is
  stored under `data/raw/drawings/<drawing-id>.png`, and the DPI value
  is recorded in a side-car (e.g., `data/raw/drawings/<drawing-id>.meta.json`)

#### Scenario: Image ingestion preserves DPI metadata
- **WHEN** a real-plan image is ingested without a PDF source
- **THEN** the pipeline reads the EXIF or PIL DPI tag and records it
  in the side-car; if no DPI is declared, the configured `INPUT_DPI`
  is used and that fact is recorded

### Requirement: Synthetic-generation tile geometry invariant

The synthetic generator SHALL emit training tiles at `TILE_SIZE = 1280`
pixels square with `TILE_STEP = 1080` pixels (200-pixel overlap). All
column-sizing constants MUST be calibrated to the post-tile pixel scale
so a column rendered at training time has the same per-tile pixel size
it would have at inference time on a real plan tiled with the same
geometry.

#### Scenario: Training and inference tile sizes match
- **WHEN** the training pipeline reads tiles and the inference
  pipeline tiles a real plan
- **THEN** both use `TILE_SIZE = 1280` and `TILE_STEP = 1080`, and the
  YOLO `imgsz` argument equals `TILE_SIZE`

#### Scenario: Changing tile geometry without recalibrating column sizes is rejected
- **WHEN** a contributor changes `TILE_SIZE` or `TILE_STEP` in
  `generate_column.py` without changing the column-size constants
- **THEN** code review flags the change as a baseline-breaking
  regression

### Requirement: No column blocking by late synthetic drawers

Synthetic generation MUST NOT paint ink over any rectangle in
`col_rects`. Every drawer in `generate_image` that runs after the
column-placement phase SHALL gate every drawn element against
`col_rects` and skip placements that would overlap.

#### Scenario: Late drawer attempts to paint over a column
- **WHEN** any drawer (e.g., `draw_filled_triangle_markers`,
  `draw_extra_bubbles`, `draw_slab_signs`, `draw_grid_crossing_decoys`)
  computes a candidate AABB
- **THEN** it MUST check `_bbox_overlaps_any(aabb, col_rects)` before
  drawing, and skip (or retry) on overlap

#### Scenario: New drawer added later
- **WHEN** a contributor adds a new late drawer that paints ink
- **THEN** the drawer MUST accept `col_rects` as a parameter, gate
  every element against it, and append the drawn AABB to `col_rects`
  so subsequent drawers also avoid it

### Requirement: Per-tile centre-in-tile labelling without dedup

The tile saver SHALL emit a column label in EVERY tile whose extent
contains the column's centre. Duplicate labels in the 200-pixel overlap
zones are required behaviour; cross-tile dedup MUST NOT be performed
at training time.

#### Scenario: Column centre in overlap zone
- **WHEN** a column's centre lies within the overlap region of two
  adjacent tiles
- **THEN** the column SHALL be labelled in BOTH tiles, with normalised
  coordinates in each tile-local space

#### Scenario: Single owning-tile assignment is rejected
- **WHEN** a contributor proposes a "closest-tile-centre" owning-tile
  rule at training time
- **THEN** the proposal MUST be rejected; cross-tile dedup belongs
  only at inference NMS

### Requirement: Stratified per-drawing train/val/test split

The data pipeline SHALL partition drawings into train, val, and test
splits keyed on **drawing ID**, never on individual bounding boxes.
Boxes from a single drawing MUST appear in exactly one split.

#### Scenario: Drawing assignment to split
- **WHEN** the splitter runs on `data/raw/drawings/`
- **THEN** each drawing ID appears in exactly one of
  `data/splits/{train,val,test}.txt`, and no bbox file under
  `data/raw/labels/<drawing-id>/` is shared between splits

#### Scenario: New drawing added to corpus
- **WHEN** a new drawing is ingested
- **THEN** it is assigned to a split deterministically (e.g., via a
  hash of the drawing ID modulo split ratios) so the assignment is
  reproducible across runs

### Requirement: Hard-negative pool seeded from past false positives

The data pipeline SHALL maintain a hard-negative pool under
`data/hard_negatives/` that the next training cycle consumes as
background examples. Entries MUST originate from corrections labelled
as FP via the human review interface; entries MUST NOT include any
real column bbox.

#### Scenario: FP correction seeds the pool
- **WHEN** a correction with `is_delete = 1` is logged against a
  detection at drawing `<drawing-id>`, bbox `(x1, y1, x2, y2)`
- **THEN** a background crop (padded by a configurable margin) is
  written to `data/hard_negatives/<drawing-id>__<hash>.png` and
  registered in a pool manifest

#### Scenario: Pool entries enter the next training run
- **WHEN** the next training cycle starts
- **THEN** every entry in the pool manifest is included as a
  zero-label image in the train split, raising the model's
  background score on that region

### Requirement: Bare-stair / bare-lift synthetic variants

The synthetic generator SHALL render approximately 50% of stair
instances and 50% of lift instances WITHOUT corner T/+ columns AND
WITHOUT edge flanking columns. The bare branches MUST instead place a
few unlabelled small outlined-rect decoys at the outer corners so the
model learns "stair / lift corner outline ≠ column".

#### Scenario: Bare-stair roll
- **WHEN** the per-stair random roll falls in the bare-stair branch
- **THEN** neither `_place_outer_corner_tp_columns` nor
  `_place_flanking_with_budget` is called, and
  `_place_unlabelled_corner_negatives` is called instead

#### Scenario: Bare-rate distribution
- **WHEN** the generator is run over a large batch (200+ stairs)
- **THEN** approximately 50% (±5%) of stairs are bare, verifiable by
  hooking the placer functions

### Requirement: Tight YOLO label bbox with 1-pixel margin

Saved YOLO label bboxes SHALL inscribe the actual drawn column extent
with a fixed `LABEL_PAD = 1` pixel margin on each side. The same
constant MUST be consumed by `_yolo_label`, `_padded_rect`, and
`_is_orphan_label`.

#### Scenario: Default label production
- **WHEN** `place_column` returns a shape bbox
- **THEN** `_yolo_label(bbox, cls)` produces a YOLO label with width
  `actual_width + 2` and height `actual_height + 2`

#### Scenario: Single source of truth
- **WHEN** any of `_yolo_label`, `_padded_rect`, or
  `_is_orphan_label` reference label padding
- **THEN** they MUST use the single module-level `LABEL_PAD` constant
