## ADDED Requirements

### Requirement: Trainable rescue YOLO proposer

The detection model SHALL include a second YOLO weight artefact
`column_rescue.pt` (yolo11n initial baseline, ~2.6 M params, single-
class) at the repository root. This rescue model SHALL run at
inference time in parallel with the frozen `column_detect.pt` on
every tile. Its outputs SHALL be merged into the same set of
candidates passed to the post-inference filtering pipeline.

`column_rescue.pt` is the ONLY weight that retrains in the HITL
loop. `column_detect.pt` MUST remain frozen forever; the rescue
training cycle MUST NOT read, write, or copy `column_detect.pt`.

#### Scenario: Rescue model runs on every tile
- **WHEN** inference is invoked on a tile
- **THEN** both `column_detect.pt` and `column_rescue.pt` produce
  predictions on the same tile input
- **AND** their outputs are concatenated before the post-process
  pipeline runs

#### Scenario: Frozen baseline is preserved across rescue training
- **WHEN** `scripts/train_yolo_rescue.py` completes a training cycle
- **THEN** the SHA / mtime of `column_detect.pt` is unchanged from
  before the cycle
- **AND** the cycle's output is written to `column_rescue.pt` only

#### Scenario: Rescue proposes a missed column
- **WHEN** a tile contains a column not detected by
  `column_detect.pt`
- **AND** that tile is present in `data/rescue_tiles/` with a
  positive YOLO-format label
- **AND** `column_rescue.pt` has been retrained on the pool
  containing that tile and passed the absorption gate
- **THEN** `column_rescue.pt` SHALL propose a bbox at that location
- **AND** the union step SHALL include that proposal in the input
  to the post-process pipeline

#### Scenario: Rescue suppresses a known-FP region
- **WHEN** a tile is present in `data/rescue_tiles/` with an empty
  `.txt` label file at a location where YOLO previously over-fired
- **AND** `column_rescue.pt` has been retrained on the pool
  containing that tile
- **THEN** `column_rescue.pt` SHALL emit zero proposals at IoU ≥
  `τ_fp` against that location

### Requirement: Union-of-detectors step with source tagging

The inference path SHALL union `column_detect.pt` and
`column_rescue.pt` predictions via a single cross-detector NMS
pass before any other filter runs. Every surviving prediction
SHALL carry a string field `source` with value `detect`,
`rescue`, or `both` (the latter when an NMS suppression collapsed
one-from-each into one surviving box).

The union NMS threshold SHALL be configurable; default is the same
0.15 IoU as the post-process IoU-NMS stage.

#### Scenario: Both detectors fire at the same location
- **WHEN** `column_detect.pt` and `column_rescue.pt` both predict
  bboxes whose IoU exceeds the union NMS threshold
- **THEN** the union step retains exactly one surviving prediction
- **AND** the surviving prediction's `source` field is `both`

#### Scenario: Only the rescue model fires
- **WHEN** `column_rescue.pt` predicts a bbox at a location where
  `column_detect.pt` predicted nothing
- **THEN** the surviving prediction's `source` field is `rescue`

#### Scenario: Only the baseline fires
- **WHEN** `column_detect.pt` predicts a bbox at a location where
  `column_rescue.pt` predicted nothing
- **THEN** the surviving prediction's `source` field is `detect`

### Requirement: Rescue model graceful degradation

When `column_rescue.pt` is missing from the repository root, the
inference path SHALL fall back to `column_detect.pt`-only output
without crashing. A diagnostic SHALL be printed once per process
naming the missing path and the resulting fallback. Inference
SHALL continue.

This soft-fail is the rollback safety net: removing the rescue
weights produces the pre-change behaviour, not an exception.

#### Scenario: Rescue weights absent
- **WHEN** inference is invoked and `column_rescue.pt` is not
  present at the configured path
- **THEN** the call returns `column_detect.pt`-only predictions
- **AND** stderr / log contains the diagnostic
  `[inference] column_rescue.pt missing → main-detector-only`
- **AND** the call does NOT raise

#### Scenario: Rescue weights present but corrupt
- **WHEN** `column_rescue.pt` exists but cannot be loaded (corrupt
  state_dict, version mismatch, etc.)
- **THEN** the inference call falls back to baseline-only output
  with a printed diagnostic
- **AND** the call does NOT raise

### Requirement: `meta.json` carries `rescue_version`

The per-job `data/jobs/<id>/px_detections.json` SHALL include a
`meta.rescue_version` field (mtime epoch seconds of
`column_rescue.pt`, or `null` when the rescue weights are absent).
This field SHALL participate in the post-process pipeline's
`@memory_first` cache key composition. The `meta` block SHALL NOT
contain a `classifier_version` field.

#### Scenario: meta.rescue_version present after inference
- **WHEN** an inference call writes `px_detections.json`
- **AND** `column_rescue.pt` exists
- **THEN** `meta.rescue_version` equals
  `os.path.getmtime("column_rescue.pt")`

#### Scenario: meta.rescue_version null on fallback
- **WHEN** an inference call writes `px_detections.json`
- **AND** `column_rescue.pt` does NOT exist
- **THEN** `meta.rescue_version` is `null`

#### Scenario: Cache key composition
- **WHEN** the post-process pipeline computes a `@memory_first`
  cache key for a job
- **THEN** the key incorporates `meta.rescue_version`
- **AND** the key does NOT incorporate any classifier version

## MODIFIED Requirements

### Requirement: Post-inference filtering pipeline

The inference path SHALL apply a fixed-order post-inference
pipeline against the **union of `column_detect.pt` and
`column_rescue.pt` predictions** before returning final detections:

(0) **Union of detectors** — concatenate `column_detect.pt` and
`column_rescue.pt` predictions per tile, then apply a single
cross-detector NMS pass at the configured union threshold
(default IoU `0.15`), tagging each survivor's `source` field.

(1) aspect filter (drop `max(w,h)/min(w,h) > 2.0`)

(2) size filter (drop sides outside `[12, 60]` px)

(3) shape filter (require ≥40 % fill OR ≥35 % border-ring dark)

(4) OCR text filter (drop bboxes where Tesseract reads ≥2
alphanumeric chars at conf ≥50)

(5) centre-distance NMS (drop within `50` px of higher-conf
detection)

(6) IoU NMS at `0.15`

The optional stair-mask MAY be enabled via the `USE_STAIR_MASK`
toggle; default `False`.

**No classifier-veto stage** SHALL exist between OCR and centre-NMS
or anywhere else in this pipeline. The rescue YOLO's missing-label
training at FP locations performs that role end-to-end.

#### Scenario: Pipeline order is fixed
- **WHEN** the inference path runs on a non-empty raw prediction set
- **THEN** stage (0) union-of-detectors runs first, followed by
  stages (1) through (6) in the listed order
- **AND** the per-stage drop counts MUST be logged

#### Scenario: Empty raw input
- **WHEN** both detectors emit zero raw predictions
- **THEN** the pipeline short-circuits and returns
  `boxes_final = np.zeros((0, 4))` without raising

#### Scenario: OCR unavailable
- **WHEN** `pytesseract` is not installed at inference time
- **THEN** filter (4) is skipped with a printed notice; the rest
  of the pipeline continues

#### Scenario: No classifier import remains
- **WHEN** `scripts/postprocess_pipeline.py` is imported
- **THEN** no module under `column_review.bbox_classifier`,
  `column_classifier`, or any classifier-filter symbol is imported
- **AND** `PostprocessConfig` has no `use_classifier_filter`,
  `classifier_weights`, or `classifier_threshold` field
