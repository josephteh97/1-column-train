## ADDED Requirements

### Requirement: Unified `rescue_tiles/` correction pool

The HITL correction signal SHALL be persisted as a single on-disk
pool under `data/rescue_tiles/` containing 1280×1280 JPG tiles in
`images/` and YOLO-format `.txt` label files in `labels/`. A
`manifest.json` SHALL record per-entry `(filename, drawing_id,
job_id, source_correction_ids[], timestamp, kind)` where `kind` is
`fn_positive` (positive label present) or `fp_negative` (empty
label file).

Positive entries (`kind=fn_positive`) SHALL include positive YOLO
label lines for every accepted positive bbox falling inside the
tile bounds (the source FN_ADDED bbox itself, plus every
`tp_confirmations` row whose centre falls in the tile, plus every
model-source detection in the tile that is NOT marked FP and NOT
`source="human_added"`).

Negative entries (`kind=fp_negative`) SHALL have an empty `.txt`
label file. The empty label is YOLO's standard "no column at this
location" supervision.

No separate `hard_negatives/` or `fn_positives/` directories
SHALL exist after the migration. The pool SHALL be the single
source of truth for correction-driven training data.

The pool MUST survive ⌫ Clear detections: once an entry is on
disk, no UI action other than its corresponding correction being
rescinded in `corrections.db` SHALL remove it.

#### Scenario: FN_ADDED writes a positive tile
- **WHEN** a correction row with `is_delete=0` and
  `changes.source='human_added'` is committed to `corrections.db`
- **AND** `scripts/rescue_tile_pool.py` runs
- **THEN** `data/rescue_tiles/images/<drawing>__<hash>.jpg` exists
- **AND** the matching `labels/<drawing>__<hash>.txt` contains at
  least the FN_ADDED bbox as a YOLO label line
- **AND** `manifest.json["entries"]` contains an entry with
  `kind="fn_positive"` referencing that correction's id

#### Scenario: FP writes a negative tile
- **WHEN** a correction row with `is_delete=1` (not rescinded by
  a later delete_fn) is committed to `corrections.db`
- **AND** `scripts/rescue_tile_pool.py` runs
- **THEN** `data/rescue_tiles/images/<drawing>__<hash>.jpg` exists
- **AND** the matching `labels/<drawing>__<hash>.txt` is empty
- **AND** `manifest.json["entries"]` contains an entry with
  `kind="fp_negative"` referencing that correction's id

#### Scenario: Tile coordinate collision hard-fails
- **WHEN** a new correction would write a tile at coordinates
  already represented in the pool with a different `kind`
- **THEN** `rescue_tile_pool.py` raises a structured exception
  naming the existing entry's id and the conflicting incoming
  correction's id
- **AND** no entry is silently overwritten

#### Scenario: Pool survives ⌫ Clear detections
- **WHEN** the user clicks ⌫ Clear detections for a job after
  the absorption gate releases it
- **THEN** every entry in `data/rescue_tiles/manifest.json` whose
  source correction was for that job remains on disk
- **AND** `corrections.db` rows for the job are cleared

#### Scenario: Rescind removes the entry
- **WHEN** a correction row is rescinded (the rescind invariant of
  `iter_effective_corrections` masks it)
- **AND** `scripts/rescue_tile_pool.py` re-runs
- **THEN** the corresponding pool entry is unlinked from disk
- **AND** `manifest.json` no longer references it

### Requirement: Single-model absorption gate

After every rescue training cycle, an automatic absorption gate
SHALL evaluate the new `column_rescue.pt` against every correction
in the latest batch BEFORE the new weights are published. The gate
SHALL hard-fail (refuse publication) if either FN coverage OR FP
suppression fails its threshold.

**FN coverage check**: for every effective `is_delete=0`
correction, the new `column_rescue.pt` SHALL propose at least one
bbox with IoU ≥ `τ_fn` against the FN_ADDED bbox. Default
`τ_fn = 0.5`.

**FP suppression check**: for every effective `is_delete=1`
correction, the new `column_rescue.pt` SHALL emit zero proposals
with IoU ≥ `τ_fp` against the FP bbox. Default `τ_fp = 0.3`.

On gate failure: the quarantined weights SHALL be moved to
`column_rescue_quarantine_<timestamp>.pt`, the published path
`column_rescue.pt` SHALL remain at its previous value, and a
structured diagnostic SHALL be written to
`column_rescue.meta.json["gate_failure"]` listing every failing
correction id, its bbox, and its IoU against the closest rescue
proposal.

The thresholds `τ_fn` and `τ_fp` SHALL be configurable from
`config.yaml`.

#### Scenario: FN absorption check passes
- **WHEN** a rescue training cycle completes
- **AND** for every is_delete=0 correction in the latest batch,
  the new weights propose a bbox at IoU ≥ τ_fn
- **AND** for every is_delete=1 correction in the latest batch,
  the new weights emit zero proposals at IoU ≥ τ_fp
- **THEN** the new weights are published to `column_rescue.pt`
- **AND** `column_rescue.meta.json` records
  `gate_status="passed"`

#### Scenario: FN absorption check fails
- **WHEN** a rescue training cycle completes
- **AND** at least one is_delete=0 correction has no matching
  rescue proposal at IoU ≥ τ_fn
- **THEN** the weights are moved to
  `column_rescue_quarantine_<ts>.pt`
- **AND** `column_rescue.pt` is NOT overwritten
- **AND** the structured diagnostic in
  `column_rescue.meta.json["gate_failure"]` names the failing
  correction ids and their bboxes
- **AND** the training subprocess exits with non-zero status

#### Scenario: FP suppression check fails
- **WHEN** a rescue training cycle completes
- **AND** at least one is_delete=1 correction still has a rescue
  proposal at IoU ≥ τ_fp
- **THEN** the weights are quarantined, the diagnostic names the
  failing FP correction ids, the published path is unchanged

#### Scenario: Gate failure surfaces to UI
- **WHEN** the absorption gate writes a `gate_failure` block
- **AND** the HITL UI polls `/api/jobs/latest`
- **THEN** the UI displays the structured diagnostic
- **AND** the UI does NOT advance to the "training succeeded"
  state

### Requirement: ⌫ Clear detections absorption gate (HTTP 412)

`POST /api/detections/clear` SHALL refuse with HTTP 412 when
`corrections.db` has any row for `job_id` whose timestamp exceeds
`column_rescue.meta.json["latest_correction_ts_per_job"][job_id]`.
The 412 detail body SHALL include:

- `error = "corrections_not_absorbed"`
- `n_uncovered` (count of uncovered correction rows)
- `last_train_ts` (the value read from meta.json, or `0`)
- `max_corr_ts` (the max correction timestamp for the job)
- `hint` (human-readable explanation pointing at 🧠 Train Rescue)

A missing or unreadable `column_rescue.meta.json` SHALL be
treated as `last_train_ts = 0` (everything is uncovered).

#### Scenario: Clear blocked before first training
- **WHEN** `column_rescue.meta.json` does not exist
- **AND** `corrections.db` has at least one row for `job_id`
- **AND** the user invokes `POST /api/detections/clear`
- **THEN** the response is HTTP 412 with
  `detail.error = "corrections_not_absorbed"` and
  `detail.last_train_ts = 0`

#### Scenario: Clear blocked with stale coverage
- **WHEN** `column_rescue.meta.json["latest_correction_ts_per_job"][job_id]`
  is set but at least one row in `corrections.db` for that job
  has a newer timestamp
- **THEN** the response is HTTP 412 with `detail.n_uncovered`
  matching the count of newer rows

#### Scenario: Clear succeeds after fresh training
- **WHEN** the user clicks 🧠 Train Rescue, training completes,
  the absorption gate publishes new weights, and meta.json
  records the updated timestamps
- **AND** the user invokes `POST /api/detections/clear`
- **THEN** the response is HTTP 200 and the job state is wiped

## MODIFIED Requirements

### Requirement: False positives feed the rescue training pool

Every correction with `is_delete = 1` (not rescinded) MUST be
absorbed into the unified `data/rescue_tiles/` pool as a
`kind="fp_negative"` entry (empty `.txt` label) before the next
training cycle. The absorption is automatic: the rescue training
entry point invokes `scripts/rescue_tile_pool.py` before starting
training. That script reads `corrections.db`, locates the FP bbox
in `data/jobs/{job_id}/px_detections.json`, crops the
1280×1280 tile surrounding it from `data/jobs/{job_id}/render.jpg`,
and writes the tile + empty `.txt` label.

No separate `data/hard_negatives/` directory SHALL exist.

#### Scenario: FP becomes empty-label tile in next training cycle
- **WHEN** the next rescue retrain runs after at least one FP row
  exists
- **THEN** the FP bbox's surrounding tile is present in
  `data/rescue_tiles/images/` with an empty matching `.txt` in
  `labels/`
- **AND** the manifest entry has `kind="fp_negative"`

### Requirement: False negatives become positive training labels

The rescue training cycle SHALL emit a positive YOLO label for
every effective correction with `is_delete = 0` and
`changes.source = 'human_added'`. The bbox in `changes.bbox` MUST
be the source of truth. The label SHALL be written as part of the
tile's `.txt` label file under `data/rescue_tiles/labels/`,
alongside every other accepted positive whose centre falls in the
same tile (TP confirmations + un-FP'd implicit TPs).

The training cycle SHALL NOT write to any `data/fn_positives/` or
similar directory — the unified `rescue_tiles/` pool is the only
positive-label sink.

#### Scenario: FN add becomes a YOLO label in a rescue tile
- **WHEN** the next rescue retrain runs after a human-added FN row
  exists
- **THEN** a tile under `data/rescue_tiles/images/` contains the
  FN, and the matching `.txt` label includes the line
  `0 cx cy bw bh` (normalised) for that FN's bbox

#### Scenario: All accepted positives in a tile are labelled
- **WHEN** a positive rescue tile is assembled around an FN_ADDED
- **AND** the same tile contains N other accepted positives
  (TP confirmations or un-FP'd model detections)
- **THEN** the tile's `.txt` label file contains N+1 YOLO label
  lines

### Requirement: Bbox edits become updated positive labels

The rescue training cycle SHALL use the corrected bbox as the YOLO
label whenever an effective correction has `is_delete = 0` and a
`changes.bbox` present (but no `source = 'human_added'`). The
original bbox MUST be retained in `original_element` for audit,
not used as a label. The corrected bbox SHALL appear in the
matching rescue-tile `.txt` label file.

#### Scenario: Edit-then-retrain
- **WHEN** a reviewer edits a bbox via `record_edit(job_id, i,
  new_bbox)` and the next rescue retrain runs
- **THEN** the rescue tile containing the edited bbox uses
  `new_bbox` (not the original) in its `.txt` label

### Requirement: Per-revision metrics recorded after every retrain

Every rescue training cycle SHALL produce a metrics snapshot
written to `data/metrics/<revision>.json` containing at minimum
the following fields, all computed against the test split:

- `mAP50` (post-union output)
- `mAP50_95` (post-union output)
- `precision`
- `recall`
- `fp_rate_per_drawing`
- `revision`
- `n_corrections_consumed` (deletes + edits + adds since previous
  revision)
- `n_rescue_tiles` (size of the unified pool at retrain time,
  partitioned by `kind`)
- `target_model` (`column_rescue.pt`)
- `gate_status` (`passed` or `failed`)

When `gate_status="failed"`, the metrics file SHALL also include
`gate_failure_summary` linking to the structured diagnostic.

#### Scenario: Metrics emitted after retrain
- **WHEN** `scripts/train_yolo_rescue.py` completes (gate passed
  OR failed)
- **THEN** `data/metrics/<timestamp>.json` exists with all the
  listed fields populated

#### Scenario: Metrics persist across retrains
- **WHEN** five rescue training cycles have completed
- **THEN** five JSON files exist under `data/metrics/`, named by
  their revision

### Requirement: TGCH-TD-S-200-L3-00 regression test

The `data/metrics/<revision>.json` for every rescue retrain SHALL
include a sub-object `regression.tgch_td_s_200_l3_00` with
detection-count and recall against the labelled ground truth of
440 column instances on drawing `TGCH-TD-S-200-L3-00` (composed
of 387 C2 + 53 C9, counted as 440 instances for the single-class
detector). Recall MUST be evaluated against the **post-union
output of the two-YOLO pipeline**, not against `column_rescue.pt`
alone. The regression is ONE test, not the full eval set.

#### Scenario: Regression sub-object present
- **WHEN** any rescue retrain completes
- **THEN** `regression.tgch_td_s_200_l3_00.expected = 440`,
  `regression.tgch_td_s_200_l3_00.detected = N`, and
  `regression.tgch_td_s_200_l3_00.recall = N / 440` are populated

#### Scenario: Regression evaluates the union, not the rescue alone
- **WHEN** the regression is computed
- **THEN** the prediction set MUST be the post-union output
  (`column_detect.pt` ∪ `column_rescue.pt`) after the full
  post-process pipeline

### Requirement: Auditable retrain provenance

Every `data/metrics/<revision>.json` MUST link back to the inputs
that produced the retrain, specifically: the SHA / mtime of the
COCO-pretrained `yolo11n.pt` used as base init, the SHAs of every
`data/jobs/{job_id}/` directory consumed, the SHA of
`data/rescue_tiles/` at retrain time, and the SHA of the
`config.yaml` block governing `τ_fn` and `τ_fp`.

#### Scenario: Audit chain is reproducible
- **WHEN** the auditor inspects `data/metrics/<revision>.json`
- **THEN** the listed input SHAs identify exactly the files that
  produced the retrain; re-running with the same inputs reproduces
  the same metrics within stochastic variance

### Requirement: Manual promotion gate

The HITL retrain loop MUST NOT auto-promote weights to
`column_detect.pt` under any circumstance. Rescue weights flow
through TWO gates before they become live:

1. **Automatic absorption gate** (required) — refuses publication
   to `column_rescue.pt` unless FN coverage and FP suppression
   pass.
2. **Manual review of `data/metrics/<revision>.json`** — the
   reviewer inspects metrics vs. the prior revision before
   accepting the cycle's output.

`column_detect.pt` SHALL NEVER be overwritten by any HITL
training cycle.

#### Scenario: Reviewer inspects before declaring a cycle accepted
- **WHEN** `scripts/train_yolo_rescue.py` finishes and the
  absorption gate publishes new `column_rescue.pt`
- **THEN** `column_detect.pt` is unchanged
- **AND** the reviewer has the option to revert by restoring the
  previous `column_rescue.pt` from `archive/`

#### Scenario: column_detect.pt is never written
- **WHEN** any HITL training script (current or future) runs
- **THEN** the SHA / mtime of `column_detect.pt` is identical
  before and after the run

## REMOVED Requirements

### Requirement: False positives feed the hard-negative pool

**Reason:** Replaced by the unified `rescue_tiles/` pool (see the
new "False positives feed the rescue training pool" requirement
above). The separate `data/hard_negatives/` directory and its 64×64
crop format are gone; FP corrections are now encoded as empty
`.txt` label files alongside positive tiles in
`data/rescue_tiles/`, which YOLO consumes directly as standard
missing-label negative supervision. The dual-pool split was an
artifact of the CNN classifier requiring crop-shaped inputs;
without the classifier, one tile-shaped pool suffices.

**Migration:** `scripts/migrate_pools_to_rescue_tiles.py` reads
`data/hard_negatives/manifest.json`, locates each crop's surrounding
1280×1280 tile in `data/jobs/<id>/render.jpg`, and writes the tile
plus an empty `.txt` label to `data/rescue_tiles/`. The original
`data/hard_negatives/` directory is archived to
`archive/pre-rescue-yolo/hard_negatives/` for one release cycle,
then deleted.
