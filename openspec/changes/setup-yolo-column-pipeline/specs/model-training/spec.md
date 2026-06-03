## ADDED Requirements

### Requirement: Train on synthetic tiles at the calibrated tile size

The trainer SHALL load the synthetic YOLO dataset under `dataset/column/`
and train YOLOv11s with `imgsz = TILE_SIZE = 1280`. Training MUST NOT
randomly resize tiles such that columns shift outside their calibrated
pixel-size band.

#### Scenario: Default training run
- **WHEN** `python3 train.py` is invoked with no arguments
- **THEN** the YOLO training call uses `imgsz=1280`, reads `dataset/column/data.yaml`, and emits checkpoints to `runs/detect/column_detector/weights/`

### Requirement: Architectural-drawing augmentation policy

The trainer MUST disable rotation, shear, perspective, and hue-rotation augmentations because architectural plans are axis-aligned grayscale line drawings. Concretely: `degrees=0`, `shear=0`, `perspective=0`, `hsv_h=0` SHALL be set. HSV saturation/value adjustments and mosaic are permitted.

#### Scenario: Aug config validation
- **WHEN** `train.py` constructs the YOLO `train(...)` kwargs
- **THEN** the kwargs SHALL include `degrees=0`, `shear=0`, `perspective=0`, and `hsv_h=0`; any later contributor adding rotation augmentations MUST justify the change

### Requirement: Manual promotion of trained weights

The trainer SHALL copy `runs/detect/<run>/weights/best.pt` to
`column_detect.pt` at the repo root only as a deliberate post-training
step. Continue-training MUST NOT auto-overwrite the baseline.

#### Scenario: Successful training
- **WHEN** a training run completes successfully
- **THEN** `train.py` SHALL copy `best.pt` to `column_detect.pt` at the repo root

#### Scenario: Continue-training output
- **WHEN** `train_continue.py` produces a fine-tuned weight
- **THEN** the output SHALL be written to `column_detect_continued.pt`, NOT to `column_detect.pt`; promotion of the continued weight to the baseline file SHALL be a manual `cp` step

### Requirement: Finalize step picks up best.pt after interruption

`finalize.py` SHALL allow the user to recover the best weight after a
training run was Ctrl-C'd post-convergence. It SHALL copy `best.pt` to
`column_detect.pt` and run the same evaluation step the trainer would
have run on a clean exit.

#### Scenario: Training interrupted after convergence
- **WHEN** the user Ctrl-C's `train.py` after mAP has plateaued and then runs `python3 finalize.py`
- **THEN** `column_detect.pt` is updated with `best.pt` and the evaluation summary is printed

### Requirement: BatchNorm-safe batch size

The training batch size SHALL satisfy `batch >= 4` so BatchNorm statistics
are well-defined. The default `batch=4` was tuned for 8 GB VRAM; lowering
it without compensating (e.g. with SyncBatchNorm or accumulation) is
forbidden.

#### Scenario: Train kwargs constructed
- **WHEN** `train.py` builds its training arguments
- **THEN** `batch` SHALL be at least `4`
