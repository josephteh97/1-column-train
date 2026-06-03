## ADDED Requirements

### Requirement: Run inference and present detections with overlay

The review interface SHALL run the detection model on a single drawing,
overlay the surviving detection bounding boxes on the source image, and
present each detection to the reviewer in a navigable view (e.g., a
paginated thumbnail grid with bbox-highlighted crops). The full plan
MUST also be saved to disk as `data/jobs/{job_id}/render.jpg` so the
review state is reproducible.

#### Scenario: Job is registered on inference
- **WHEN** the reviewer runs the interface against a real plan
- **THEN** a fresh `job_id` is generated, `data/jobs/{job_id}/render.jpg`
  is written with the full plan, and
  `data/jobs/{job_id}/px_detections.json` is written with the
  post-processed detections in `{"columns": [{"bbox":[x1,y1,x2,y2],
  "score": float, ...}, ...]}` form

#### Scenario: Per-detection presentation
- **WHEN** the reviewer opens the review view for a registered job
- **THEN** every detection in `px_detections.json["columns"]` is
  surfaced as a crop with its red bbox visible, indexed by its position
  in the JSON list

### Requirement: Per-detection mark as TP / FP / missed

The reviewer SHALL be able to mark each presented detection as TP
(keep), FP (delete), or supply additional MISSED (FN) detections via
explicit coordinates. Marks MUST be persisted only when the reviewer
explicitly saves; per-click writes are forbidden.

#### Scenario: FP mark and save
- **WHEN** the reviewer flips detection index `i` to DELETE and
  presses the Save button
- **THEN** one row is appended to `data/corrections.db` with
  `element_index = i`, `is_delete = 1`, `element_type = 'column'`, and
  `original_element` is the JSON of the original detection entry

#### Scenario: Idempotent save
- **WHEN** the reviewer presses Save twice in the same session without
  changing any mark
- **THEN** the second save does NOT produce duplicate rows in
  `corrections.db` for the same `(job_id, element_index)`

#### Scenario: FN add
- **WHEN** the reviewer supplies a missed column as `(cx, cy, size_px)`
- **THEN** a synthetic bbox is computed, appended to
  `data/jobs/{job_id}/px_detections.json`, and one row is appended to
  `corrections.db` with `is_delete = 0`, `element_index` pointing at
  the new entry, and `changes.source = 'human_added'`

### Requirement: Persistence schema

The interface SHALL persist all reviewer marks via the
`scripts/corrections_logger.py` module to the following schema:

```
data/corrections.db                     — SQLite
data/jobs/{job_id}/render.jpg           — the plan as reviewed
data/jobs/{job_id}/px_detections.json   — { "columns": [...] }
```

The `corrections` table MUST have columns `(id, job_id, element_type,
element_index, original_element JSON, changes JSON, is_delete,
timestamp)`. `element_index` MUST index into the saved
`px_detections.json["columns"]` list at the moment the row was
written.

#### Scenario: Schema matches retrain consumer
- **WHEN** the corrections DB is consumed by `scripts/retrain_yolo.py`
- **THEN** the column names, types, and JSON shapes match exactly; no
  schema translation layer is required

### Requirement: Job state immutability after corrections exist

`data/jobs/{job_id}/px_detections.json` MUST NOT be silently
overwritten once any correction row exists for that `job_id`. Any
operation that would rewrite the file after corrections exist SHALL
refuse to run and emit a diagnostic so the reviewer can resolve
manually.

#### Scenario: Re-registering an active job is rejected
- **WHEN** `save_job(job_id, ...)` is called for a `job_id` that
  already has rows in `corrections`
- **THEN** the call MUST raise `JobAlreadyCorrected` with a message
  naming the row count and instruct the user to start a new job

### Requirement: Visual confirmation per mark

The interface SHALL render the source-image crop around every
detection so the reviewer can visually confirm what they are marking.
The crop MUST be drawn from the live source image (not a re-rendered
synthetic), with the bbox outlined.

#### Scenario: Crop matches source pixels
- **WHEN** the reviewer inspects a detection at bbox `(x1,y1,x2,y2)`
- **THEN** the crop shown is a slice of the loaded source image from
  `(x1 - pad, y1 - pad, x2 + pad, y2 + pad)` for a small fixed `pad`,
  with the bbox visibly outlined in red
