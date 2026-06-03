## 1. Ingestion-time DZI tile-pyramid generation

- [ ] 1.1 Extend `scripts/ingest_drawings.py` with a `_write_dzi(raster_path, drawing_id)` helper that emits a Microsoft DZI manifest at `data/raw/drawings/<id>.dzi` and tiles at `data/raw/drawings/<id>_files/<level>/<col>_<row>.jpg` using Pillow (tile size 256, 1-pixel overlap, JPEG quality 80, levels 0 through `floor(log2(max(W,H)))`)
- [ ] 1.2 Wire `_write_dzi` into the ingest happy path so it runs after the canonical raster is finalised and matches its pixel content; add an `--no-tiles` flag that skips it
- [ ] 1.3 Update `resolve_drawing(drawing_id)` so its returned `meta_dict` also reports `dzi_path` (or None if absent), without breaking the existing two-tuple return signature for current callers
- [ ] 1.4 Add a `python3 scripts/hitl.py build-tiles <drawing-id>` subcommand that locates the raster via `resolve_drawing`, generates the DZI tree afresh (overwriting any existing one), and is idempotent
- [ ] 1.5 Document the ~25–35 % disk-cost note in the `--help` output of `hitl.py ingest` and `hitl.py build-tiles` and in `READMD.md` near the review workflow

## 2. FastAPI backend (`scripts/correction_app/`)

- [ ] 2.1 Create `scripts/correction_app/__init__.py`, `scripts/correction_app/app.py` (FastAPI app factory), and `scripts/correction_app/static/` for the vendored OpenSeadragon + frontend assets
- [ ] 2.2 Add an idempotent sidecar-table migration that runs at app start: `CREATE TABLE IF NOT EXISTS tp_confirmations(session_id TEXT, job_id TEXT, element_index INTEGER, ts REAL, PRIMARY KEY (job_id, element_index))` and `CREATE TABLE IF NOT EXISTS reviewer_sessions(session_id TEXT PRIMARY KEY, reviewer_id TEXT NOT NULL, started_ts REAL NOT NULL)` — no ALTER on any existing table
- [ ] 2.3 Implement `GET /drawings/{drawing_id}` returning JSON `{ dzi_url, raster_size: [W,H], detections: <px_detections.json contents>, job_id, session_id }`; create the `job_id` via `corrections_logger.new_job_id()` and persist the render+detections via `save_job` if not yet present
- [ ] 2.4 Implement `POST /drawings/{drawing_id}/marks` accepting `{ kind: "TP"|"FP"|"FN_ADDED"|"DELETE_FN"|"RESCIND_FP", element_index?, bbox?, session_id }` — route TP to `tp_confirmations`, FP to `corrections_logger.record_delete`, FN_ADDED to `corrections_logger.record_add`, and rescinds via the existing `record_edit` flow; commit synchronously and return only after fsync via `os.replace` so the 1 s durability requirement holds
- [ ] 2.5 Implement `POST /drawings/{drawing_id}/marks/batch` for batch-mark-FP and batch-delete actions, applying each in one transaction
- [ ] 2.6 Implement `GET /drawings/{drawing_id}/state` returning the consolidated four-state map for the drawing so the frontend can restore on reload (reads `corrections` for FP + rescind, `tp_confirmations` for TP, `px_detections.json` for FN_ADDED via the `human_added` source key)
- [ ] 2.7 Serve `data/raw/drawings/<id>.dzi` and `<id>_files/` as static routes scoped to `<id>`, with a HEAD probe endpoint `/drawings/{drawing_id}/dzi-exists` for the load-time check
- [ ] 2.8 Bind to 127.0.0.1 on a default port 8765; pick the next free port if busy and print it; accept `--tile-cache-mb` and `--hit-tolerance-px` and `--snap-grid-px` and pass them through to the frontend as JSON config at page load

## 3. OpenSeadragon frontend (`scripts/correction_app/static/`)

- [ ] 3.1 Vendor OpenSeadragon (single JS bundle) under `static/vendor/`; write `index.html` that instantiates an OSD viewer over the served DZI tile source and lays out the main viewport, mini-map container, progress UI strip, and zoom-level indicator
- [ ] 3.2 Implement a single custom canvas overlay layer that maps detection bounding boxes from world coordinates to screen coordinates via the OSD `update-viewport` event, with bbox culling against the visible viewport so off-screen boxes are not drawn
- [ ] 3.3 Implement the four-state colour + shape-or-border treatment so each of unreviewed/TP/FP/FN_ADDED is unambiguously distinguishable at high zoom, low zoom, and on the mini-map; pass WCAG AA contrast on both white and black backgrounds
- [ ] 3.4 Implement zoom-adaptive hit-test: on left-click, compute the world-space tolerance from `max(hit_tolerance_css_px, hit_tolerance_css_px / zoom_factor)` and pick the nearest detection within tolerance
- [ ] 3.5 Bind primary actions to single-key shortcuts: T (TP), F (FP), D (delete), A (begin add-FN drag), U (undo), Y (redo), N (next unreviewed), P (previous unreviewed), `+`/`-` (zoom on cursor or selection), Space (pan), `0` (100%), Esc (clear selection), the configured fit-to-window key; reject any modal dialogs anywhere in the marking loop
- [ ] 3.6 Implement the 100-deep undo/redo ring buffer with inverse-op records; ensure push/pop/re-apply are O(1) per action and every action issues a server round-trip whose return is awaited before the action shows as saved
- [ ] 3.7 Implement single-fluid-drag FN add: A begins, the next mouse-down + drag + mouse-up commits one bbox via `POST /drawings/.../marks` with kind=FN_ADDED; support optional snap-to-grid
- [ ] 3.8 Implement rubber-band select (Shift+drag), batch-mark-FP, batch-delete-FN — all via `POST /drawings/.../marks/batch`
- [ ] 3.9 Implement the mini-map (down-scaled OSD navigator) with unreviewed-cluster highlights driven by the four-state map; fit-to-window, 100%, zoom-to-selection, reset-view shortcuts; clickable numeric zoom-level indicator with type-an-exact-percent input
- [ ] 3.10 Implement live progress counts, jump-to-next-unreviewed action (N), previous-unreviewed action (P), and filter-by-state mode that hides three of the four state categories
- [ ] 3.11 Implement the load-time performance probe: HEAD-check DZI, run a synthetic 2000-box overlay render once and measure, time level-0 tile fetch — any failure shows a single full-screen diagnostic banner and blocks marking
- [ ] 3.12 Implement the first-launch reviewer-id prompt (single non-modal inline input) that writes to `~/.column-review.json` on submit and calls a `POST /session` to insert the `reviewer_sessions` row

## 4. CLI wiring in `scripts/hitl.py`

- [ ] 4.1 Add a `review <drawing-id>` subcommand to `scripts/hitl.py` that launches the FastAPI app (`uvicorn` via subprocess or `app.run`) on 127.0.0.1:8765 (or next free port), opens the browser via `webbrowser.open`, and blocks on the server process
- [ ] 4.2 Rewrite `cmd_ingest`'s post-success print-hint and `cmd_status`'s zero-corrections hint so they direct the reviewer to `python3 scripts/hitl.py review <drawing-id>` instead of the deleted notebook
- [ ] 4.3 Update `READMD.md` so the worked-example for TGCH-TD-S-200-L3-00 replaces the "Open correct_detections.ipynb" step with the `hitl.py review` invocation, and update the workflow-at-a-glance and when-to-run-which-file sections accordingly
- [ ] 4.4 Update `CLAUDE.md` so the project-purpose and pipeline diagrams drop the `correct_detections.ipynb` reference and name the new web app instead

## 5. Delete the old UI and verify the preserved contract

- [ ] 5.1 Delete `correct_detections.ipynb` from the repository root
- [ ] 5.2 Grep the codebase for the substrings "correct_detections", "ipynb", "notebook" and confirm only documentation-history references remain (no executable code path still names the notebook); remove any orphaned references
- [ ] 5.3 Run `python3 scripts/hitl.py retrain --dry-run` after one round of writes by the new UI on a test drawing and confirm `scripts/retrain_yolo.py` reads the new FP rows, the existing rescind logic, and FN_ADDED entries without modification
- [ ] 5.4 Run `python3 scripts/hard_negative_pool.py --dry-run` after the same round and confirm it consumes the new FP rows via `iter_effective_corrections` without modification
- [ ] 5.5 Run `openspec validate rebuild-correction-ui-web` and confirm the change reports valid
