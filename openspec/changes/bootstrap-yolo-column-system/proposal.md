## Why

Structural columns must be located on 2D architectural and structural
floor plans for downstream BIM / quantity workflows. Off-the-shelf
detection models are blind to architectural symbols and cannot be
audited; the existing `setup-yolo-column-pipeline` change codified a
synthetic-only YOLO pipeline but did not specify real-data ingestion,
out-of-distribution failure behaviour, the human review interface as
a first-class component, or per-revision auditing. This change
bootstraps the full system — data, model, review UI, feedback loop —
as one auditable specification.

## What Changes

- **SUPERSEDES** the prior change `setup-yolo-column-pipeline`. Its
  three capabilities (`synthetic-data-generation`, `model-training`,
  `inference-post-processing`) are absorbed into the four new
  capabilities listed below. The 7-filter inference pipeline becomes a
  required behaviour of the detection model, not a parallel capability.
- Establish **single-class scope**: one class `column` (class id `0`)
  regardless of column type, geometry, or material (square RC,
  rectangular RC, circular RC, composite steel-concrete, steel I/H,
  hollow sections, project-specific variants). Type / family /
  dimension classification is OUT OF SCOPE and handled by a separate
  downstream module.
- Add **real-data ingestion** to the data pipeline: ingest annotated
  floor-plan PDFs and images alongside synthetic generation, with
  stratified train/val/test splits keyed on drawing ID (no bounding
  boxes from the same drawing leak across splits).
- Add a **hard-negative pool** seeded from past false positives that
  the next training cycle consumes as background examples.
- Add **explicit OOD hard-failure behaviour** to the detection model:
  no silent fallback. When input DPI or detection density falls
  outside a calibrated band, the system aborts with a diagnostic
  rather than emitting low-quality predictions.
- Add a **configurable confidence threshold** and **configurable input
  DPI** as the two public knobs of the detection model.
- Add the **human review interface** as a first-class system
  component: runs inference on a drawing, presents detections with
  bounding-box overlay, lets the reviewer mark each as TP / FP /
  missed (FN), and persists the marks against drawing ID and image
  coordinates.
- Add the **feedback loop** that closes the cycle: FPs → hard-negative
  pool, FNs → new positive labels after human verification, with
  per-revision metrics (mAP@0.5, mAP@0.5:0.95, precision, recall, FP
  rate per drawing) recorded so each retrain iteration is auditable.
- Name a **regression benchmark**: drawing TGCH-TD-S-200-L3-00 with
  440 column instances (composed of 387 C2 + 53 C9, counted as 440
  instances for this single-class detector). This is one regression
  test, not the full evaluation set.

## Capabilities

### New Capabilities

- `data-pipeline`: Ingest annotated real floor-plan PDFs and images at
  a configurable DPI, generate synthetic training tiles from drawing
  templates, partition into stratified train/val/test splits by
  drawing ID, and maintain a hard-negative pool seeded from past false
  positives. Subsumes the prior `synthetic-data-generation`
  capability's invariants (TILE_SIZE = 1280, no column blocking, per-
  tile centre-in-tile labelling without dedup, bare-stair/lift
  variants, LABEL_PAD).
- `detection-model`: A YOLOv11s single-class detector trained on the
  data pipeline's output, with one configurable confidence threshold,
  configurable input DPI, manual promotion of `column_detect.pt`, the
  7-filter post-inference pipeline (aspect, size, shape, OCR-text,
  centre-NMS, IoU-NMS, optional stair-mask) as a required behaviour,
  and explicit OOD hard-failure. Subsumes the prior `model-training`
  and `inference-post-processing` capabilities.
- `human-review-interface`: A program that runs inference on a single
  drawing, presents detections with bounding-box overlay, and lets a
  reviewer mark each detection as TP, FP, or missed (FN). Persists
  marks tied to drawing ID and image coordinates in a structured
  format (`data/corrections.db` + `data/jobs/{job_id}/`) consumable by
  the feedback loop.
- `feedback-loop`: Absorbs reviewer marks into the next training
  cycle: FPs become hard-negative training samples, FNs become
  positive samples after human verification. Tracks per-revision
  metrics (mAP@0.5, mAP@0.5:0.95, precision, recall, FP rate per
  drawing) so each retrain iteration is auditable. Names
  TGCH-TD-S-200-L3-00 (440 instances) as one regression test.

### Modified Capabilities

(none — this change is the bootstrap; the prior change's capabilities
are superseded rather than modified in-place.)

## Impact

- **Code**: builds on the existing `generate_column.py`,
  `train.py`/`train_continue.py`/`finalize.py`, `test_column.ipynb`,
  `scripts/postprocess_detections.py`, `scripts/corrections_logger.py`,
  `scripts/retrain_yolo.py`, and `correct_detections.ipynb`. Net-new
  code: real-PDF/image ingestion module, per-drawing split utility,
  hard-negative-pool manager, OOD detector, per-revision metrics
  recorder.
- **Deployed artifact**: `column_detect.pt` at the repo root remains
  the single deployed weight. Promotion is manual (`cp`); the trainer
  never auto-overwrites the baseline.
- **Data**:
  - `data/raw/{drawings,labels}/` — new ingestion location for real
    annotated plans.
  - `data/synthetic/` — generated from `generate_column.py`.
  - `data/splits/{train,val,test}.txt` — per-drawing split manifests.
  - `data/hard_negatives/` — FP-seeded background tiles.
  - `data/corrections.db` + `data/jobs/{job_id}/{render.jpg,px_detections.json}`
    — HITL persistence.
  - `data/metrics/{revision}.json` — per-retrain metric snapshots.
- **Dependencies**: ultralytics, torch, Pillow, numpy, cv2 (existing).
  New: pdfplumber or pdf2image for PDF rasterisation. pytesseract +
  tesseract binary (already installed) for the text-FP filter step.
  ipywidgets (already installed) for the review UI v1.
- **Non-goals (explicit)**: column type / family / dimension
  classification; Revit element output; beam / slab detection;
  modifying the existing v3/v4 Revit C# add-in; multi-class extension;
  any online-learning path that bypasses the corrections loop.
