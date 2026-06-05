## Why

The current detection cascade is `column_detect.pt` (frozen yolo11s) →
post-process pipeline → `column_classifier.pt` (98k-param CNN veto).
The CNN classifier is structurally a *patch classifier*, not a
detector — it has an encoder but no decoder, takes 64×64 crops as
input, and can only reject proposals. It cannot recover columns that
`column_detect.pt` never proposed, so the project's stated objective
of "detect columns the baseline missed" is unreachable via the
current trainable component. False-negative recovery is structurally
gated by the CNN's lack of a proposal mechanism.

Replacing the CNN with a second, *trainable* YOLO (`column_rescue.pt`,
yolo11n) gives the system a real detector with proposal capability.
The frozen baseline is untouched; the rescue model alone absorbs
every reviewer correction and is the only component that retrains.
A unified `rescue_tiles/` pool replaces the two separate
`hard_negatives/` and `fn_positives/` pools — YOLO's training
convention (positive labels for FNs, empty label files for FPs)
expresses both signals in one storage format. An absorption gate
runs after every retrain to verify the new weights actually learned
the corrections before publication.

## What Changes

- **BREAKING** Remove the CNN classifier veto stage from the
  inference cascade. `column_classifier.pt`, `bbox_classifier.py`,
  `train_bbox_classifier.py`, and the `classifier_filter` stage in
  `postprocess_pipeline.py` are deleted. `column_classifier.pt` is
  archived to `archive/pre-rescue-yolo/` for one release cycle.
- **BREAKING** Remove the dual correction pool schema
  (`data/hard_negatives/` + the just-added `data/fn_positives/`).
  A one-shot migration script consolidates surviving FP-crop data
  into `data/rescue_tiles/` as 1280×1280 tiles + YOLO labels.
- **BREAKING** Remove the ~30-second per-click CNN retrain
  expectation. Rescue YOLO retrain is ~20 minutes per cycle on the
  target GPU and is intended to run per-batch (after a HITL
  session), not per-click.
- Add `column_rescue.pt` (yolo11n, ~2.6 M params) as a second,
  trainable YOLO running in parallel with the frozen
  `column_detect.pt`. Outputs are unioned via NMS before the
  geometry pipeline, with each surviving proposal tagged
  `detect` / `rescue` / `both` for telemetry.
- Add `data/rescue_tiles/{images,labels}/` as the single unified
  correction pool. Positive `.txt` labels encode FN_ADDED bboxes;
  empty `.txt` labels encode FP regions (label-absent negative
  supervision). Tile coordinate collisions hard-fail rather than
  silently overwrite.
- Add the single-model absorption gate. After every retrain, the
  gate verifies the new `column_rescue.pt` proposes a bbox at
  IoU ≥ τ_fn for every FN_ADDED in the latest correction batch AND
  emits zero proposals at IoU ≥ τ_fp for every FP region. Failure
  on either criterion refuses publication, archives the weights to
  a quarantine path, and surfaces a structured diagnostic to the
  HITL UI.
- Add a `rescue_version` field to the column-detection `meta.json`
  for cache-key composition. The previous `classifier_version`
  field is removed.
- Rename the HITL retrain control "🧠 Train CNN" → "🧠 Train
  Rescue", surface epoch counter + ETA progress, and lock the
  control while training is in progress.

## Capabilities

### New Capabilities

(none — this change modifies existing capabilities established by
`bootstrap-yolo-column-system`.)

### Modified Capabilities

- `detection-model`: switches the inference cascade from "frozen
  YOLO → pipeline → CNN classifier veto" to "frozen YOLO + trainable
  rescue YOLO → union(NMS) → pipeline → out". Adds the
  trainable-rescue-proposer requirement; modifies the pipeline-flow
  requirement; removes the classifier-veto-stage requirement; adds
  `rescue_version` to the `meta.json` schema and removes
  `classifier_version`.
- `feedback-loop`: replaces the dual hard-negative + FN-positive
  crop pools with a single unified `rescue_tiles/` tile + label
  pool whose contents survive ⌫ Clear detections; adds the
  single-model absorption gate that hard-fails publication when the
  retrained rescue weights have not learned the latest correction
  batch; removes the dual-pool schema requirement.
- `human-review-interface`: renames the retrain control from
  "🧠 Train CNN" to "🧠 Train Rescue"; adds progress display
  (epoch + ETA) and double-click lockout because retrains take
  ~20 minutes; surfaces absorption-gate failures to the user;
  removes the fast-retrain-loop requirement.

## Impact

- **Code (deleted)**: `column_review/bbox_classifier.py`,
  `scripts/train_bbox_classifier.py`,
  `scripts/hard_negative_pool.py`,
  `scripts/fn_positive_pool.py` (created earlier this session,
  never wired up). Classifier-filter stage removed from
  `scripts/postprocess_pipeline.py` and `column_review/inference.py`.
- **Code (new)**: `column_review/yolo_rescue.py` (load + predict
  helper, mtime-keyed cache); `scripts/rescue_tile_pool.py`
  (assembles `data/rescue_tiles/` from `corrections.db`);
  `scripts/train_yolo_rescue.py` (trains yolo11n on the unified
  pool + synthetic dataset); `scripts/absorption_gate.py` (post-
  retrain FN / FP coverage check); `scripts/migrate_pools_to_rescue_tiles.py`
  (one-shot consolidation of surviving FP-crop data).
- **Code (modified)**: `column_review/inference.py`,
  `scripts/postprocess_pipeline.py`,
  `column_review/routes/train.py`,
  `column_review/routes/detections.py`,
  `column_review/static/app.js`, `CLAUDE.md`.
- **Deployed artifacts**:
  - `column_detect.pt` (existing) — unchanged, frozen forever.
  - `column_rescue.pt` (NEW) — written by the rescue training
    script, gated by the absorption check before publication.
  - `column_classifier.pt` (EXISTING) — archived to
    `archive/pre-rescue-yolo/column_classifier.pt` for one release
    cycle, then deleted.
- **Data layout**:
  - NEW: `data/rescue_tiles/{images,labels}/` (1280×1280 JPGs +
    YOLO `.txt` labels), `data/rescue_tiles/manifest.json`.
  - REMOVED: `data/hard_negatives/`, `data/fn_positives/` (both
    archived during migration, then deleted).
  - `data/corrections.db` schema unchanged.
- **API surface**: HTTP endpoint `/api/train-classifier` is renamed
  `/api/train-rescue`. `POST /api/detections/clear` adds the
  HTTP 412 absorption-gate gate. No other route changes.
- **Dependencies**: no new packages. `ultralytics`, `torch`, `PIL`,
  `cv2`, `numpy` (all already present).
- **Performance**: per-tile inference cost roughly doubles (now
  runs two YOLOs sequentially). On the target GPU, this is
  acceptable. Train Rescue takes ~20 minutes vs. ~30 seconds for
  the deleted Train CNN; the per-batch cadence covers it.
- **Non-goals (explicit)**: no parallel inference of the two YOLOs
  (sequential per tile is fine for now); no fast-path back to
  CNN-classifier-only mode; no migration of `corrections.db` rows
  (the schema is unchanged); no per-class extension (single-class
  scope preserved).
