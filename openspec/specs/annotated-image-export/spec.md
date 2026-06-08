# annotated-image-export Specification

## Purpose

Provide a one-click way for reviewers to export the current detections
rendered onto the full-resolution source A0 image, with a provenance
footer identifying which model artifacts produced the boxes. The
export is read-only with respect to all training/correction state so
it can run safely at any time, including while a `Train YOLO2+CNN`
job is in progress.

## Requirements

### Requirement: Toolbar export control

The `column_review` UI SHALL expose an "Export Annotated Image"
control in the toolbar that triggers server-side rendering of the
current detections onto the source A0 image at full resolution.

The control SHALL be hidden until a drawing is open and detections are
available — matching the gating used by `infer-btn`, `clear-det-btn`,
and `train-both-btn` (the `show-after-open hidden` class).

#### Scenario: User clicks Export Annotated Image with detections present

- **WHEN** a drawing is open and detections are visible on the canvas
- **AND** the user clicks "Export Annotated Image"
- **THEN** the UI SHALL POST `{session_id}` to `/api/export-annotated`
- **AND** while the request is in flight the UI SHALL show an inline
  "Exporting…" indicator on or beside the button
- **AND** on a `200` response the UI SHALL surface the returned `path`
  as inline text the reviewer can read and copy

#### Scenario: User clicks Export Annotated Image before opening a drawing

- **WHEN** no drawing is open
- **THEN** the button SHALL be hidden (covered by the
  `show-after-open hidden` class set on every toolbar action that
  requires an open drawing)
- **AND** the user SHALL NOT be able to trigger the export

#### Scenario: Export endpoint returns an error

- **WHEN** `/api/export-annotated` returns a non-2xx status
- **THEN** the UI SHALL surface the response error inline to the user
- **AND** the UI SHALL re-enable the button so the reviewer can retry

### Requirement: Full-resolution server-side render

The backend SHALL render the current detections for the requested
session onto the source A0 image at the source's native resolution,
using Pillow, with no downsampling.

The render SHALL:
- Open the source image from `data/raw/drawings/<drawing_id>.<ext>`
  (where `<ext>` ∈ `{png, jpg}`, resolved by file presence).
- Read the most-recent detections for the session through the same
  data path that powers the canvas overlay (no separate inference
  run is permitted at export time).
- Draw each bbox as a rectangle stroke (default colour, 4 px wide)
  with a label using the Pillow default font.
- Write the result as a PNG to `output/<drawing_id>_annotated_<unix_ts>.png`
  where `<unix_ts>` is `int(time.time())` at request handling.

#### Scenario: Source image present and detections available

- **WHEN** `POST /api/export-annotated` is called with a valid `session_id`
- **AND** `data/raw/drawings/<drawing_id>.<ext>` exists
- **AND** the session has detections
- **THEN** the endpoint SHALL produce a PNG at
  `output/<drawing_id>_annotated_<unix_ts>.png`
- **AND** the PNG dimensions SHALL equal the source image dimensions
- **AND** the response SHALL be `{ok: true, path: "output/<file>.png"}`

#### Scenario: Source image missing

- **WHEN** `POST /api/export-annotated` is called
- **AND** no file matches `data/raw/drawings/<drawing_id>.{png,jpg}`
- **THEN** the endpoint SHALL respond `404` with a structured error
  identifying the missing source path
- **AND** no file SHALL be written to `output/`

#### Scenario: Re-inference is NOT performed at export time

- **WHEN** `POST /api/export-annotated` is called
- **THEN** the endpoint SHALL NOT invoke `run_pipeline` or any model
  forward pass
- **AND** the rendered bboxes SHALL be byte-equivalent to the
  detections currently held in the session store

#### Scenario: Session is invalid

- **WHEN** `POST /api/export-annotated` is called with a `session_id`
  that fails `validate_session(...)`
- **THEN** the endpoint SHALL respond with the same error contract as
  every other route that calls `validate_session` (HTTP `404` /
  structured error body)
- **AND** no file SHALL be written to `output/`

### Requirement: Provenance footer

The rendered PNG SHALL carry a single-line footer at the bottom of the
image stating, at minimum, the version (or `mtime`) of each model in
the cascade and the export timestamp.

The footer SHALL be rendered into the image pixels — not as a sidecar
file, EXIF tag, or PNG metadata chunk — so that the artifact remains
self-documenting after copy, paste, or re-encoding.

#### Scenario: All meta.json files present

- **WHEN** `POST /api/export-annotated` runs
- **AND** `column_classifier.meta.json` and `column_rescue.meta.json`
  both exist at the repo root
- **AND** `column_detect.pt` exists at the repo root
- **THEN** the footer SHALL include:
    - `column_detect.pt` mtime (formatted as a local timestamp)
    - `column_rescue` `saved_ts` (= last promotion through the
      absorption gate), `gate_status`, and `epochs_trained` from
      `column_rescue.meta.json`
    - `column_classifier` `saved_ts` (= last training cycle),
      `best_val_acc`, and `epochs_trained` from
      `column_classifier.meta.json`
    - the export timestamp (`int(time.time())` formatted as local time)

#### Scenario: A meta.json file is missing

- **WHEN** `POST /api/export-annotated` runs
- **AND** one or more of `column_classifier.meta.json` /
  `column_rescue.meta.json` is absent
- **THEN** the footer SHALL render `absent` in place of the missing
  field
- **AND** the export SHALL succeed
- **AND** the response SHALL still be `{ok: true, path: ...}`

#### Scenario: Footer is legible on a light background

- **WHEN** the footer is rendered
- **THEN** the background of the footer band SHALL contrast with the
  source line drawing (e.g. translucent dark band behind light text)
- **AND** the footer SHALL be readable when the image is viewed at
  1:1 zoom in a standard image viewer

### Requirement: Output location and filename

Exports SHALL be written to the repo-root `output/` directory using
the filename `<drawing_id>_annotated_<unix_ts>.png`.

The endpoint SHALL ensure the `output/` directory exists (create on
first export) before writing.

#### Scenario: output/ does not yet exist

- **WHEN** the endpoint runs and `output/` does not exist
- **THEN** the endpoint SHALL create `output/`
- **AND** the export SHALL proceed without error

#### Scenario: Two exports requested in the same second

- **WHEN** two `POST /api/export-annotated` calls land in the same
  wall-clock second for the same drawing
- **THEN** the second call MAY overwrite the first call's PNG (the
  `int(time.time())` filename is the same)
- **AND** the endpoint SHALL NOT raise an error; the second
  reviewer's export takes precedence

### Requirement: Read-only with respect to detection and correction data

The export endpoint SHALL be read-only with respect to:
- `column_detect.pt`, `column_rescue.pt`, `column_classifier.pt`
- `column_classifier.meta.json`, `column_rescue.meta.json`
- `data/corrections.db`
- `data/jobs/<id>/`
- the absorption gate state

The endpoint SHALL NOT trigger training, ⌫ Clear, retraining, or any
write to the corrections store.

#### Scenario: Export runs concurrently with a Train YOLO2+CNN job

- **WHEN** a `Train YOLO2+CNN` job is in progress
- **AND** a reviewer clicks "Export Annotated Image"
- **THEN** the export SHALL proceed independently using the currently
  loaded model artifacts
- **AND** the export SHALL NOT block, queue against, or interact with
  the retrain job's state machine

#### Scenario: Export does not touch corrections.db

- **WHEN** `POST /api/export-annotated` runs to completion
- **THEN** `data/corrections.db` mtime SHALL be unchanged
- **AND** no row SHALL be inserted, updated, or deleted in any
  `corrections.db` table
