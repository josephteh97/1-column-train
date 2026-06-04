## Why

The correction UI shipped under `scripts/correction_app/` is non-functional
in practice: when a floor plan is opened, the raster and the model
detections never both render together. Because the image and the detection
boxes are never visible on screen simultaneously, the reviewer cannot click
on a detection to mark it, cannot drag-add a missed column, cannot save
corrections, and therefore cannot kick off retraining. The entire
human-in-the-loop correction-and-retraining flywheel is blocked behind a
blank page. Patching the existing app has been tried; the page still fails
the basic "image + detections both visible" contract for the reviewer.

A wholesale replacement is needed: a single-command web UI that is
**ground-up redesigned**, optimised for comfort over thousands of
corrections per session on A0 (300 DPI) floor plans, and that hard-fails
loudly if the renderer ends up in the "blank canvas" state that motivated
the rewrite.

## What Changes

- **NEW** top-level Python package `column_review/` with a `column-review`
  CLI entry-point (`pip install -e .` then run from any directory).
- **NEW** single-command launch: `column-review` starts a local FastAPI
  server on a free port (auto-picks one if the default is taken), opens
  the browser to the file picker, no multi-step setup, no frontend build.
- **NEW** pyramidal/tile-based rendering of A0 @ 300 DPI floor plans
  (OpenSeadragon over the DZI pyramid that `scripts/hitl.py ingest`
  already produces). Open-to-first-render under 3s; 60 fps pan/zoom; LRU
  tile cache capped at 512 MB.
- **NEW** R2 regression guard: a frontend canary checks both the image
  and the detections rendered, and raises a visible fail banner if either
  side is missing. The blank-page failure mode that prompted this change
  MUST be detectable, never silent.
- **NEW** keyboard-first correction loop: every primary action has a
  single-key shortcut; the full mark/undo/redo/save-and-submit workflow
  completes without touching a menu or modal.
- **NEW** Save & Submit action triggers `scripts/retrain_yolo.py` as a
  background `subprocess.Popen`, after a single confirm dialog (outside
  the correction loop, so the "no modals in the loop" rule holds).
  Retraining status (queued / running / completed / failed) is surfaced
  in the UI without blocking review work.
- **BREAKING**: the entire `scripts/correction_app/` package (8 files,
  ~2,440 LOC) is deleted in this change. The `review` subcommand of
  `scripts/hitl.py` is removed. No legacy mode, no feature flag, no
  transition period.
- **BREAKING**: `README.md` and `CLAUDE.md` user-facing workflow lines
  that invoke `python3 scripts/hitl.py review <drawing-id>` are
  rewritten in this same change to invoke `column-review` instead.
- Doc comments in `scripts/corrections_logger.py`,
  `scripts/postprocess_pipeline.py`, and `scripts/ingest_drawings.py`
  that point at `scripts/correction_app/` are updated to point at
  `column_review/`.
- `data/corrections.db` schema is preserved (`corrections`,
  `tp_confirmations`, `reviewer_sessions` tables). The new package
  reuses `scripts/corrections_logger.py` as the canonical SQLite layer
  and takes ownership of the sidecar CREATE TABLE statements that the
  deleted `correction_app/app.py` used to run at startup.
- `scripts/hitl.py` retains `ingest`, `build-tiles`, `retrain`, and
  `status` subcommands. Only `review` is removed.
- `scripts/ingest_drawings.py`, `scripts/retrain_yolo.py`, the YOLO
  weights file `column_detect.pt`, and the DZI tile pyramid layout
  (`data/raw/drawings/<id>.dzi` + `<id>_files/`) are unchanged.

## Capabilities

### New Capabilities

- `correction-ui`: Reviewer-facing web tool that, given a floor-plan
  image with a pre-built DZI tile pyramid and a `column_detect.pt`
  inference path, lets a single reviewer mark model detections as
  False Positive, drag-add missed columns as False Negatives, and
  trigger a retraining job. Covers the five-step user workflow
  (launch → open → review/correct → save & submit → retrain) and the
  twelve hard requirements (single-command launch, image+detections
  co-rendering with regression guard, A0 tile-pyramid pan/zoom,
  keyboard-first input, mouse interactions, four visual states,
  navigation aids, zoom-adaptive hit-testing, ≥100-level undo/redo,
  autosave-on-action, performance ceiling, retrain trigger contract).

### Modified Capabilities

<!-- None. `openspec/specs/` is currently empty; no archived capabilities
     to amend. The prior `rebuild-correction-ui-web` change has not been
     archived and is superseded by this change. -->

## Impact

- **Deleted code**: `scripts/correction_app/` (8 files, ~2,440 LOC)
  including `app.py`, `static/app.js`, `static/index.html`,
  `static/styles.css`, `static/vendor/openseadragon.min.js`, and the
  package `__init__.py`.
- **Deleted CLI surface**: `python3 scripts/hitl.py review <drawing-id>`
  is removed. The `cmd_review` handler and `review` subparser in
  `scripts/hitl.py` (~70 LOC) go away.
- **New code**: `column_review/` package (~12 files: `cli.py`,
  `server.py`, `inference.py`, `db.py`, `retrain_jobs.py`, four
  `routes/` modules, `static/index.html`, `static/app.js`,
  `static/styles.css`, `static/vendor/openseadragon.min.js`).
- **New CLI surface**: `column-review` console_script registered in
  `pyproject.toml` via `[project.scripts]`.
- **Updated user-facing docs**: `README.md` (5 mentions of the old
  command) and `CLAUDE.md` (2 architectural references) are rewritten
  in this change.
- **Shared infrastructure** (NOT changed):
  `scripts/corrections_logger.py`, `scripts/retrain_yolo.py`,
  `scripts/ingest_drawings.py`, `scripts/postprocess_pipeline.py`,
  the `data/corrections.db` schema, the `data/raw/drawings/<id>_files/`
  DZI tile-pyramid layout, the `column_detect.pt` deployed model, and
  `scripts/hitl.py {ingest, build-tiles, retrain, status}` subcommands.
- **Downstream consumers**:
  - `scripts/retrain_yolo.py` reads `data/corrections.db` exactly as
    before — no contract change.
  - `data/raw/drawings/<id>_files/` consumers are: only the deleted
    `scripts/correction_app/` (gone) and the new `column_review/`
    package (reads, never writes).
  - No script or notebook in the repo imports `scripts.correction_app`
    as a Python module; the only references are doc comments (updated
    in this change) and the user-facing CLI line in README/CLAUDE.md
    (also updated in this change). Silent breakage is therefore
    precluded.
- **Out of scope** (explicit non-goals from the proposal prompt):
  multi-reviewer collaboration; comment threads / audit logs / history
  beyond undo-redo; OCR; any UI surface beyond the correction loop
  (inference dashboards, training monitors, ingestion UI); cloud
  deployment; authentication; migration tooling for the broken page's
  data (no real reviewer corrections exist in `data/corrections.db`
  from the old page); archiving the prior `rebuild-correction-ui-web`
  change directory (separate `/opsx:archive` follow-up).
