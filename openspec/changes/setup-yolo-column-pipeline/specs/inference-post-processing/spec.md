## ADDED Requirements

### Requirement: Tile real plan at the training tile geometry

The inference path SHALL tile a real-plan input image with the same geometry
used at training: `TILE_SIZE = 1280`, `TILE_STEP = 1080` (200-px overlap).
Per-tile YOLO predictions SHALL be translated from tile-local to global
canvas coordinates before any cross-tile processing.

#### Scenario: Plan larger than one tile
- **WHEN** the inference pipeline receives a plan image larger than `TILE_SIZE`
- **THEN** it SHALL produce a grid of overlapping tiles stepping by `TILE_STEP`, run YOLO on each, and accumulate detections in global image coordinates

### Requirement: Six-filter post-processing pipeline

The post-processing pipeline SHALL execute six filters in fixed order against the raw per-tile predictions; each filter MUST remove a specific FP class and pass survivors to the next filter:

1. **STAIR-MASK** — detect stair regions via parallel-line clustering
   (`cv2.HoughLinesP` with 12-35 px tread spacing, ≥3 treads, ±3°
   axis alignment) and drop any detection whose centre is inside a
   detected stair region.
2. **ASPECT** — drop detections with `max(w,h)/min(w,h) > 2.0`.
3. **SIZE** — drop detections outside `[12, 60]` px per side.
4. **SHAPE** — drop detections unless EITHER the bbox fill ratio ≥
   `0.40` (filled-column path) OR the fixed 2-px border ring dark
   ratio ≥ `0.35` (outlined-column path).
5. **CENTRE-NMS** — greedy dedup: any two detections with centres
   within `50` pixels of an already-kept higher-confidence detection
   are dropped.
6. **IoU-NMS BACKUP** — final torchvision NMS at `iou_thr = 0.15` for
   partial overlaps that survived step 5.

#### Scenario: All six filters execute
- **WHEN** non-empty raw detections enter the pipeline
- **THEN** the cell prints the per-step survivor count and produces `boxes_final` / `scores_final` with the final survivors

#### Scenario: Empty raw input
- **WHEN** `all_boxes` is empty (model emits zero raw detections)
- **THEN** the pipeline SHALL short-circuit, set `boxes_final = np.zeros((0, 4))` and `scores_final = np.zeros((0,))`, and NOT attempt to run the filters (avoids `tvops.nms` shape error)

### Requirement: Stair-region detector graceful degradation

If `cv2` is unavailable at runtime, the stair detector SHALL return an
empty list without raising. The rest of the pipeline (filters 1-5)
continues to execute.

#### Scenario: cv2 not installed
- **WHEN** `_detect_stair_regions` runs in an environment without `cv2`
- **THEN** the function prints a notice and returns `[]`; no stair-mask filtering happens but no exception is raised

### Requirement: Importable script module

The post-processing logic SHALL be available as both an importable
module (`scripts/postprocess_detections.py`) for downstream consumers
AND a notebook cell (`test_column.ipynb` cell 5) for interactive QA.

#### Scenario: Downstream consumer imports
- **WHEN** an external tool imports `from scripts.postprocess_detections import filter_detections`
- **THEN** the import resolves and `filter_detections(detections)` accepts the documented detection-dict format and returns the filtered list

### Requirement: Empty detection list passes through cleanly

Every pipeline stage MUST accept an empty input list and SHALL emit an empty output list without raising. This applies to both the script-module functions (`aspect_filter`, `center_distance_nms`, `cross_tile_nms`) and the cell-5 inline steps.

#### Scenario: Empty input to every stage
- **WHEN** the input list to `aspect_filter`, `center_distance_nms`, or `cross_tile_nms` is empty
- **THEN** each function SHALL return an empty list, never raise on shape mismatch
