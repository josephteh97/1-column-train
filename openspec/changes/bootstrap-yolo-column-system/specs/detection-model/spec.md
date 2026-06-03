## ADDED Requirements

### Requirement: Single-class YOLOv11s detector

The detector SHALL be a YOLOv11s model trained as single-class. All
column variants (square RC, rectangular RC, circular RC, composite
steel-concrete, steel I/H, hollow sections, project-specific) MUST
emit YOLO class id `0` with the string label `column`. The trained
weight artefact MUST be `column_detect.pt` at the repository root.

#### Scenario: Inference output class
- **WHEN** the model emits any prediction
- **THEN** the predicted class id is `0` and the class string is
  `column`

#### Scenario: Multi-class extension is rejected
- **WHEN** a contributor proposes adding a second class to this model
- **THEN** the change MUST be rejected; multi-class detection belongs
  to a separate change

### Requirement: Training imgsz matches calibrated tile size

The trainer SHALL invoke YOLO with `imgsz = TILE_SIZE = 1280` and MUST
NOT randomly resize tiles such that columns shift outside their
calibrated pixel-size band.

#### Scenario: Default training run
- **WHEN** `python3 train.py` is invoked with no arguments
- **THEN** the YOLO training call uses `imgsz=1280`, reads
  `dataset/column/data.yaml`, and emits checkpoints to
  `runs/detect/column_detector/weights/`

### Requirement: Architectural-drawing augmentation policy

The trainer MUST disable rotation, shear, perspective, and hue-rotation
augmentations because architectural plans are axis-aligned grayscale
line drawings. Specifically, `degrees=0`, `shear=0`, `perspective=0`,
and `hsv_h=0` SHALL be set.

#### Scenario: Aug config validation
- **WHEN** `train.py` constructs the YOLO `train(...)` kwargs
- **THEN** the kwargs SHALL include `degrees=0`, `shear=0`,
  `perspective=0`, and `hsv_h=0`

### Requirement: Numerical-stability training settings

The trainer SHALL set `amp=False` and `lr0=1e-3` so the run completes
without NaN losses on the synthetic distribution. Either value MUST
NOT be changed without re-running a 50-epoch stability check.

#### Scenario: Training run completes without NaN
- **WHEN** `python3 train.py` is invoked with default kwargs
- **THEN** the run reaches `epochs - 1` without raising
  `RuntimeError: Checkpoint ... is corrupted with NaN/Inf weights`

### Requirement: Manual promotion of trained weights

The trainer SHALL copy `runs/detect/<run>/weights/best.pt` to
`column_detect.pt` at the repo root only as a deliberate post-training
step. Continue-training MUST NOT auto-overwrite the baseline; its
output MUST be `column_detect_continued.pt`.

#### Scenario: Successful training
- **WHEN** a training run completes successfully
- **THEN** `train.py` SHALL copy `best.pt` to `column_detect.pt` at
  the repo root

#### Scenario: Continue-training output
- **WHEN** `train_continue.py` produces a fine-tuned weight
- **THEN** the output SHALL be written to
  `column_detect_continued.pt`, NOT to `column_detect.pt`

### Requirement: Configurable confidence threshold and input DPI

The detection model SHALL expose exactly two public inference knobs:
a confidence threshold `CONF_TH` (default `0.25`) and an input DPI
`INPUT_DPI` (default `300`). All other inference behaviour MUST be
deterministic given those two values and the loaded weight.

#### Scenario: Threshold overrides
- **WHEN** the caller invokes inference with `CONF_TH=0.40`
- **THEN** all detections with confidence `< 0.40` are excluded from
  output

#### Scenario: DPI mismatch handling
- **WHEN** a real plan is rasterised at `INPUT_DPI=300` and inference
  is run
- **THEN** tiling uses the same `INPUT_DPI` value, and the
  per-detection bbox sizes match the calibrated pixel-size band

### Requirement: Out-of-distribution hard-failure

The inference path MUST detect out-of-distribution input and abort with
a diagnostic message rather than emit silent low-quality predictions.
Two OOD signals SHALL be checked: (a) effective DPI outside a
configurable band (default `[210, 420]`, i.e., `0.7×`–`1.4×` of the
training DPI), and (b) mean per-tile raw detection count outside a
configurable band (default `[0.05, 30]`).

#### Scenario: Wrong-DPI input
- **WHEN** a plan is rasterised at `INPUT_DPI=150` (below the lower
  band)
- **THEN** inference aborts with `OutOfDistributionError: input DPI
  150 < 210` and emits zero predictions

#### Scenario: Empty-page input
- **WHEN** the mean per-tile raw detection count is below the lower
  band (e.g., the input is a blank page)
- **THEN** inference aborts with `OutOfDistributionError: mean tile
  detections N < 0.05` and emits zero predictions

### Requirement: Post-inference filtering pipeline

The inference path SHALL apply a fixed-order post-inference pipeline
against raw per-tile predictions before returning final detections:
(1) aspect filter (drop `max(w,h)/min(w,h) > 2.0`), (2) size filter
(drop sides outside `[12, 60]` px), (3) shape filter (require ≥40 %
fill OR ≥35 % border-ring dark), (4) OCR text filter (drop bboxes
where Tesseract reads ≥2 alphanumeric chars at conf ≥50), (5)
centre-distance NMS (drop within `50` px of higher-conf detection),
(6) IoU NMS at `0.15`. Filter (0) stair-mask MAY be enabled via
`USE_STAIR_MASK` toggle; default `False`.

#### Scenario: Pipeline order is fixed
- **WHEN** the inference path runs on a non-empty raw prediction set
- **THEN** the six required filters run in the listed order; the
  per-stage drop counts MUST be logged

#### Scenario: Empty raw input
- **WHEN** the model emits zero raw detections
- **THEN** the pipeline short-circuits and returns `boxes_final =
  np.zeros((0, 4))` without raising

#### Scenario: OCR unavailable
- **WHEN** `pytesseract` is not installed at inference time
- **THEN** filter (4) is skipped with a printed notice; the rest of
  the pipeline continues
