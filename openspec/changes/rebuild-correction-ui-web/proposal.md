## Why

The current correction-marking UI is `correct_detections.ipynb` — a 14-cell Jupyter notebook the reviewer steps through to mark detections as TP, FP, or add missed columns. It does not scale to the realistic workload of reviewing an A0 plan at 300 DPI (~140 megapixels, several hundred to a few thousand detection boxes): the notebook loads the full raster as a single bitmap, re-renders on every action, has no keyboard shortcuts, no undo, no mini-map, no autosave per action, and forces a separate "save" step. A reviewer cannot complete thousands of marks per session without fatigue, and the user has decided the existing approach is unacceptable. This change deletes the notebook and replaces it with a single-purpose, ground-up web-based viewer designed for high-throughput TP/FP/FN_ADDED marking on full-resolution floor plans.

## What Changes

- **BREAKING — DELETE** `correct_detections.ipynb`. The notebook is removed in this change. No flag-gated coexistence, no legacy mode, no migration period.
- **NEW** local FastAPI backend + static OpenSeadragon-based JS frontend under `scripts/correction_app/`, launched via `python3 scripts/hitl.py review <drawing-id>`. The browser opens automatically on `127.0.0.1`. Single drawing per session.
- **NEW** keyboard-first UX: T (TP), F (FP), D (delete), A (add FN drag), U/Y (undo/redo), N/P (next/previous unreviewed), `+`/`-` (zoom on cursor or selection), Space (pan), F (fit), `0` (100% / 1:1), Esc (clear selection). Every primary action has a single-key shortcut; no modal dialogs in the marking loop.
- **NEW** tile-pyramid viewer (OpenSeadragon + DZI). A0-at-300 DPI opens in under 3 seconds; pan/zoom at 60 fps regardless of zoom level. The full-resolution raster MUST NOT be loaded as a single bitmap. Tile-cache memory ceiling is configurable (default 512 MB) and enforced via LRU eviction.
- **NEW** four unambiguous visual states (unreviewed / TP / FP / FN_ADDED), each with a distinct colour PLUS minor shape/border treatment so they survive downsampling on the mini-map.
- **NEW** undo/redo with ≥100 levels, O(1) per action. Zoom-adaptive hit-test tolerance — clicking within a configurable radius selects a box; tolerance scales up at low zoom.
- **NEW** single-fluid-drag FN add (no two-step "draw then confirm"). Optional snap-to-grid.
- **NEW** batch operations: rubber-band select, mark-all-in-selection-as-FP, delete-all-in-selection.
- **NEW** progress UI: live counts of TP/FP/FN/unreviewed, mini-map with unreviewed-cluster highlights, jump-to-next-unreviewed action, filter-by-category mode, clickable numeric zoom-level indicator.
- **NEW** autosave-per-action: every mark is durable to disk within 1 second of the keystroke. No separate "save" step. Reviewer never loses work on close or crash.
- **NEW** load-time performance probe: the tool MUST hard-fail with a diagnostic message identifying the bottleneck if it cannot guarantee <50 ms interaction lag on A0 + 2000 boxes — no silent degradation.
- **ADDITIVE** ingestion-time DZI tile-pyramid generation: `scripts/ingest_drawings.py` now writes `data/raw/drawings/<id>.dzi` (Microsoft DZI XML manifest) and `data/raw/drawings/<id>_files/<level>/<col>_<row>.jpg` tiles (256 px, JPEG quality 80) per drawing alongside the existing raster + meta. Backfill subcommand `python3 scripts/hitl.py build-tiles <drawing-id>` exists for drawings already ingested. The web reviewer refuses to open a drawing whose DZI is missing — no silent fallback to single-bitmap rendering.
- **PRESERVED** `scripts/corrections_logger.py` — this file is the STORAGE layer, NOT the UI. It is imported by `scripts/retrain_yolo.py` and `scripts/hard_negative_pool.py` and MUST remain. The new FastAPI backend uses its public surface (`new_job_id`, `save_job`, `record_delete`, `record_edit`, `record_add`, `iter_effective_corrections`, `summary`) verbatim.
- **PRESERVED** the existing `corrections` SQLite schema and `data/jobs/{job_id}/px_detections.json` shape. The new UI writes rows matching the existing column layout — FP maps to `is_delete=1`, FN_ADDED appends to `px_detections.json["columns"]` with `source: "human_added"`. New UI state the existing schema cannot express (TP confirmations, reviewer session identity) is recorded in two NEW SIDECAR tables (`tp_confirmations`, `reviewer_sessions`) added by additive migration — no existing column is changed.
- **UNCHANGED** detection model, training pipeline, post-processing, retrain (`scripts/retrain_yolo.py`), hard-negative pool (`scripts/hard_negative_pool.py`), ingestion CLI ergonomics, any other UI surface. The wholesale-replacement mandate applies only to the correction-marking surface.

## Capabilities

### New Capabilities

- `correction-ui`: The web-based correction reviewer end-to-end — keyboard-first UX, tile-pyramid viewer (OpenSeadragon + DZI), four visual states, undo/redo, zoom-adaptive hit-test, single-drag FN add, batch operations, mini-map + jump-to-next-unreviewed, autosave-per-action, load-time performance probe with hard-fail, the schema-write contract that maps TP/FP/FN_ADDED onto the existing `corrections` table + sidecar `tp_confirmations` table, reviewer-id provenance via `reviewer_sessions`, and the deletion mandate (the Jupyter notebook is removed in this change; no flag-gated coexistence).
- `drawing-tile-pyramid`: The additive DZI tile-pyramid output written by `scripts/ingest_drawings.py` at ingestion time, plus the `python3 scripts/hitl.py build-tiles <drawing-id>` backfill subcommand for drawings ingested before this change.

### Modified Capabilities

(None. The existing `setup-yolo-column-pipeline` and `bootstrap-yolo-column-system` changes' capability specs are not modified — this change adds two new capabilities and leaves the others untouched.)

## Impact

**Code deleted (1 file):**
- `correct_detections.ipynb` — entire notebook removed.

**Code added:**
- `scripts/correction_app/` — FastAPI app + static OpenSeadragon frontend.
- `scripts/hitl.py` — new subcommands `review <drawing-id>` (launches the web app) and `build-tiles <drawing-id>` (backfill DZI for an existing drawing).

**Code modified (additive only):**
- `scripts/ingest_drawings.py` — write DZI alongside existing raster.
- `scripts/hitl.py` — `cmd_ingest` / `cmd_status` print-hint text rewritten to point at `hitl.py review` instead of the notebook.
- `READMD.md`, `CLAUDE.md` — notebook references replaced with web-app workflow.

**Schema (additive only — existing readers unchanged):**
- `data/corrections.db`: two new sidecar tables — `tp_confirmations(session_id, job_id, element_index, ts)` and `reviewer_sessions(session_id PRIMARY KEY, reviewer_id, started_ts)`. The existing `corrections` table is unchanged.
- `data/jobs/{job_id}/px_detections.json`: unchanged shape.

**Filesystem (additive):**
- `data/raw/drawings/<id>.dzi` + `data/raw/drawings/<id>_files/` — per drawing, ~30% disk cost on top of the raster.

**Unchanged downstream consumers** (hard contract):
- `scripts/retrain_yolo.py` — its `SELECT job_id, element_type, element_index, original_element, changes, is_delete FROM corrections` query keeps working.
- `scripts/hard_negative_pool.py` — its `iter_effective_corrections` import keeps working.
- `scripts/postprocess_pipeline.py`, training scripts, weights — untouched.

**Dependencies:**
- New runtime deps: `fastapi`, `uvicorn`, `jinja2` (optional, for templating). OpenSeadragon vendored as a single JS bundle under `scripts/correction_app/static/`.
- No new build step. The FastAPI app is a plain `python3 -m` launch.
