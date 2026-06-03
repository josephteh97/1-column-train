## ADDED Requirements

### Requirement: False positives feed the hard-negative pool

Every correction with `is_delete = 1` MUST be absorbed into the
hard-negative pool described by the `data-pipeline` capability before
the next training cycle. The absorption is automatic: the retrain
entry point reads `data/corrections.db`, locates the FP bbox in
`data/jobs/{job_id}/px_detections.json`, crops the source plan, and
writes the crop to `data/hard_negatives/`.

#### Scenario: FP appears as background in next training cycle
- **WHEN** the next retrain runs after at least one FP row exists
- **THEN** the FP bbox's source crop is present in the
  train-split dataset as a zero-label image, and the model's loss on
  it is recorded in the post-train metrics

### Requirement: False negatives become positive training labels

The retrain SHALL emit a positive YOLO label for every correction
with `is_delete = 0` and `changes.source = 'human_added'`. The bbox
in `changes.bbox` MUST be the source of truth; the entry's position
in `px_detections.json` is index-only.

#### Scenario: FN add becomes a YOLO label
- **WHEN** the next retrain runs after a human-added FN row exists
- **THEN** the retrain emits a label line `0 cx cy bw bh` (normalised)
  in the YOLO label file for the drawing, using `changes.bbox` as the
  source

### Requirement: Bbox edits become updated positive labels

The retrain SHALL use the corrected bbox as the YOLO label whenever
a correction has `is_delete = 0` and `changes.bbox` present (but no
`source = 'human_added'`). The original bbox MUST be retained in
`original_element` for audit, not used as a label.

#### Scenario: Edit-then-retrain
- **WHEN** a reviewer edits a bbox via `record_edit(job_id, i,
  new_bbox)` and the next retrain runs
- **THEN** the YOLO label for that detection uses `new_bbox`, not the
  original

### Requirement: Per-revision metrics recorded after every retrain

Every retraining cycle SHALL produce a metrics snapshot written to
`data/metrics/<revision>.json` containing at minimum the following
fields, all computed against the test split:

- `mAP50`
- `mAP50_95`
- `precision`
- `recall`
- `fp_rate_per_drawing` (per-drawing FP count divided by per-drawing
  total detections, then averaged)
- `revision` (the timestamp or commit hash of the retrain)
- `n_corrections_consumed` (deletes + edits + adds since previous
  revision)
- `n_hard_negatives` (size of the hard-negative pool at retrain time)

#### Scenario: Metrics emitted after retrain
- **WHEN** `scripts/retrain_yolo.py` completes
- **THEN** `data/metrics/<timestamp>.json` exists with all the listed
  fields populated

#### Scenario: Metrics persist across retrains
- **WHEN** five retraining cycles have completed
- **THEN** five JSON files exist under `data/metrics/`, named by their
  revision; the directory is the audit trail

### Requirement: TGCH-TD-S-200-L3-00 regression test

The `data/metrics/<revision>.json` for every retrain SHALL include a
sub-object `regression.tgch_td_s_200_l3_00` with detection-count and
recall against the labelled ground truth of 440 column instances on
drawing TGCH-TD-S-200-L3-00 (composed of 387 C2 + 53 C9, counted as
440 instances for the single-class detector). The regression is ONE
test, not the full eval set.

#### Scenario: Regression sub-object present
- **WHEN** any retrain completes
- **THEN** `regression.tgch_td_s_200_l3_00.expected = 440`,
  `regression.tgch_td_s_200_l3_00.detected = N`, and
  `regression.tgch_td_s_200_l3_00.recall = N / 440` are populated

#### Scenario: Regression is not the only signal
- **WHEN** a retrain raises `recall` on TGCH-TD-S-200-L3-00 but drops
  `mAP50` on the test split below the previous revision
- **THEN** the deployed weight is NOT auto-promoted; the human
  promoter inspects both signals and decides

### Requirement: Auditable retrain provenance

Every `data/metrics/<revision>.json` MUST link back to the inputs that
produced the retrain, specifically: the SHA of `column_detect.pt` used
as base weight, the SHAs of every `data/jobs/{job_id}/` directory
consumed, and the SHA of `data/hard_negatives/` at retrain time.

#### Scenario: Audit chain is reproducible
- **WHEN** the auditor inspects a `data/metrics/<revision>.json`
- **THEN** the listed input SHAs identify exactly the files that
  produced the retrain; re-running with the same inputs reproduces
  the same metrics within stochastic variance

### Requirement: Manual promotion gate

The feedback loop MUST NOT auto-promote the fine-tuned weight to
`column_detect.pt`. A reviewer SHALL inspect `data/metrics/<revision>.json`
against the prior revision and decide; the promotion is a manual `cp`
step.

#### Scenario: Reviewer inspects before promotion
- **WHEN** `scripts/retrain_yolo.py` finishes and produces
  `column_detect_ft_{timestamp}.pt`
- **THEN** `column_detect.pt` is unchanged; deployment requires
  `cp column_detect_ft_{timestamp}.pt column_detect.pt`
