## Why

TGCH-style architectural floor plans contain hundreds of concrete columns that
must be located, counted, and labelled for downstream BIM / cost workflows.
Manual marking is slow and error-prone, while off-the-shelf detection models
(COCO-pretrained YOLO) have never seen architectural symbols and miss most
columns. We need a self-contained, reproducible pipeline that (1) generates
sufficient synthetic training data to compensate for the lack of labelled
real plans, (2) trains a YOLOv11 detector specialised on the column class,
and (3) runs aggressive post-processing at inference time so the final
output has effectively zero false positives even on tiled real-plan input.

## What Changes

- Establish a **single-class** YOLOv11 detector (`column`) as the deployed
  artifact (`column_detect.pt` at the repo root) — produced from synthetic
  data only, validated on a held-out real plan.
- Codify the **TILE_SIZE = 1280 / TILE_STEP = 1080** invariant across data
  generation, training (`imgsz=1280`), and inference (tiled with 200-px
  overlap). Anything that breaks this calibration silently regresses
  the baseline.
- Codify the **synthetic-data quality rules** that emerged from iteration:
  - Zero tolerance for column blocking — every late drawer that paints
    ink must gate against `col_rects`.
  - Duplicate YOLO labels at tile-overlap zones are CORRECT — dedup
    belongs at inference NMS, never in `_save_tiles`.
  - Bare-stair / bare-lift variants (~30%) counter-train the
    "stair/lift edge ⇒ adjacent column" prior.
- Codify the **inference post-processing pipeline** (`test_column.ipynb`
  cell 5 + `scripts/postprocess_detections.py`) that achieves near-zero
  FP through (a) stair-region masking, (b) aspect filter, (c) size
  filter, (d) shape filter (fill OR border ratio), (e) centre-distance
  NMS, (f) IoU NMS backup.
- Provide **manual promotion workflow** (`train_continue.py`,
  `finalize.py`) so the user controls when a newly-trained weight
  replaces the deployed `column_detect.pt`.

## Capabilities

### New Capabilities

- `synthetic-data-generation`: Procedurally render A0-scale floor-plan
  canvases with labelled columns, structures (openings, stairs, lifts,
  cores), partitions, annotations, and decoy elements. Save as YOLO tiles
  with human-check overlays. Every drawer that paints ink must respect
  the no-blocking rule against existing column rectangles.
- `model-training`: Train YOLOv11s on the synthetic tile set with the
  TILE_SIZE-aligned hyperparameters. Produce `column_detect.pt` via the
  manual-promotion pattern (train → finalize → copy). Support gentle
  fine-tuning on new data (`train_continue.py`) without overwriting the
  baseline.
- `inference-post-processing`: Tile a full-resolution real plan at the
  training tile geometry, run YOLO per tile, then apply the multi-filter
  post-processing pipeline (stair mask → aspect → size → shape →
  centre-NMS → IoU-NMS) to drive FPs toward zero. Provide both an
  importable script module (`scripts/postprocess_detections.py`) and a
  notebook entry point (`test_column.ipynb`).

### Modified Capabilities

(none — first proposal)

## Impact

- **Code**: `generate_column.py` (synthetic generator), `train.py` /
  `train_continue.py` / `finalize.py` (training), `test_column.ipynb` +
  `scripts/postprocess_detections.py` (inference + post-processing).
- **Deployed artifact**: `column_detect.pt` at the repo root, consumed by
  downstream tooling. The promotion step from training output to this
  file is manual by design.
- **Data**: synthetic `dataset/column/{images,labels,human_check}/{train,val,test}/`
  rebuilt from `generate_column.py` on demand; no real-plan training labels
  required.
- **Dependencies**: ultralytics (YOLO), torch, Pillow, numpy, cv2
  (transitive via ultralytics). No new top-level deps introduced by this
  proposal.
- **Out of scope**: multi-class extension (door/wall/stairs) — reserved
  for a future change. Online learning from user corrections —
  `scripts/retrain_yolo.py` exists but the corrections-DB pipeline is
  not part of this proposal.
