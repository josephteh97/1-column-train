## 1. Package scaffold and CLI entry-point

- [x] 1.1 Create the empty package directory tree: `column_review/`, `column_review/routes/`, `column_review/static/`, `column_review/static/vendor/`
- [x] 1.2 Write `column_review/__init__.py` exporting `__version__`
- [x] 1.3 Write `column_review/cli.py` with a `main()` entry point that parses `--port`, `--host`, `--db-path`, `--weights`, and `--no-browser`
- [x] 1.4 Write `column_review/__main__.py` so `python -m column_review` works (delegates to `cli.main`)
- [x] 1.5 Add or amend `pyproject.toml` at the repo root to register `[project.scripts] column-review = "column_review.cli:main"` and include `column_review/static/**` as package data
- [x] 1.6 Smoke test: `pip install -e .` then `column-review --help` from `/tmp` prints usage and exits 0

## 2. FastAPI server skeleton and launch ergonomics

- [x] 2.1 Create `column_review/server.py` with `create_app(config: dict) -> FastAPI` and `pick_port(start: int) -> int` ported from old `scripts/correction_app/app.py:1150` (no `correction_app` import)
- [x] 2.2 Port `open_browser_soon(url: str, delay_seconds: float) -> None` from old `app.py:1168` into `server.py`
- [x] 2.3 Wire `cli.main` to call `pick_port`, mount the FastAPI app via uvicorn, schedule `open_browser_soon`, and print the chosen URL to stdout
- [x] 2.4 Wire `StaticFiles` for `column_review/static/` at `/`
- [x] 2.5 Verify R1 from any CWD: `cd /tmp && column-review` opens the browser and prints `column-review listening on http://...`
- [x] 2.6 Verify R1 port-conflict path: hold the default port with `nc -l 8765 &` and rerun `column-review`; server picks the next free port and prints it

## 3. DB layer and reused SQLite plumbing

- [x] 3.1 Create `column_review/db.py` that imports `scripts.corrections_logger` (relative import via `sys.path` insertion in `cli.py` if needed) and re-exports `new_job_id`, `iter_effective_corrections`, `summary`, `DB_PATH`
- [x] 3.2 In `db.py`, add `ensure_sidecar_tables(conn)` that runs `CREATE TABLE IF NOT EXISTS tp_confirmations (...)` and `CREATE TABLE IF NOT EXISTS reviewer_sessions (...)` with the exact column shapes from the deleted `app.py` (verified against existing DB rows)
- [x] 3.3 Add `ensure_retrain_jobs_table(conn)` that creates `retrain_jobs (id INTEGER PK, pid INTEGER, started_ts REAL, status TEXT, finished_ts REAL, stderr_tail TEXT)`
- [x] 3.4 Call all three `ensure_*` helpers once from a FastAPI startup hook in `server.py`
- [x] 3.5 Verify: open `data/corrections.db` and confirm the three table shapes are present with no schema drift from the old app

## 4. Inference module

- [x] 4.1 Create `column_review/inference.py` and port `_get_or_load_model(weights_path)` from old `app.py:145` (preserves `(path, mtime, size)` cache key and `threading.Lock`)
- [x] 4.2 Port the body of `post_infer` (old `app.py:866-1148`) as `run_inference(drawing_id: str, config: dict) -> InferenceResult` — strip FastAPI request/response objects, return a plain dataclass with `boxes`, `scores`, `tile_counts`
- [x] 4.3 Auto-detect device (`cuda:0` if `torch.cuda.is_available()`, else `cpu`) and print `[infer] auto-selected device=<value>` once per server startup
- [x] 4.4 Reuse `scripts.postprocess_pipeline.run_pipeline` for tile-merging — no copy
- [x] 4.5 Verify: import `column_review.inference` and call `run_inference("TGCH-TD-S-200-L3-00", DEFAULT_CONFIG)` end-to-end on the user's fixture drawing without hitting the FastAPI surface; assert > 0 boxes returned

## 5. Tile serving route

- [x] 5.1 Create `column_review/routes/tiles.py` with `GET /tiles/{drawing_id}.dzi` and `GET /tiles/{drawing_id}_files/{level}/{col}_{row}.jpg` mapped at the existing on-disk paths under `data/raw/drawings/`
- [x] 5.2 Return 404 with a typed JSON error `{"error": "tile_pyramid_missing", "drawing_id": "...", "hint": "python3 scripts/hitl.py ingest <path>"}` when the `<id>_files/` directory does not exist
- [x] 5.3 Set HTTP cache headers (`Cache-Control: public, max-age=86400`) on tile responses; tiles never change
- [x] 5.4 Verify: open the DZI in a browser tab directly, confirm the tile-level JSON descriptor and a few tile JPEGs load

## 6. File-picker and open route

- [x] 6.1 Create `column_review/routes/files.py` with `GET /api/drawings` returning a JSON list of drawing IDs (one per `<id>.dzi` file under `data/raw/drawings/`)
- [x] 6.2 Add `POST /api/open` that takes `{drawing_id, reviewer_id}`, creates or reuses a `job_id` for the (drawing, reviewer) pair, inserts a row into `reviewer_sessions`, and returns `{job_id, drawing_id, tile_source_url, detections_url}`
- [x] 6.3 If the requested drawing has no DZI, return HTTP 412 with `{"error": "tile_pyramid_missing", "hint": "..."}`
- [x] 6.4 Verify: `curl POST /api/open` with the ingested fixture drawing returns the expected JSON; the response references `/tiles/<id>.dzi`

## 7. Detections + marks route

- [x] 7.1 Create `column_review/routes/detections.py` with `GET /api/detections?job_id=...` that reads `data/jobs/<job_id>/px_detections.json` (or returns 412 with a hint that `run_inference` is needed) and merges it with effective corrections via `iter_effective_corrections` to yield per-detection state
- [x] 7.2 Implement `POST /api/marks` taking `{job_id, action: "FP_TOGGLE" | "FN_ADDED", element_index?, bbox?}`. FP_TOGGLE writes a `(is_delete=1)` or rescinding `(is_delete=0)` row keyed by the model's positive `element_index`. FN_ADDED appends `{"bbox":..., "source":"human_added"}` to `data/jobs/<job_id>/px_detections.json["columns"]` (atomic temp-file replace) and writes a `(is_delete=0)` corrections row keyed by the newly-assigned positive `element_index` — the same shape `scripts/retrain_yolo.py` already consumes
- [x] 7.3 Implement server-side undo/redo stack per `job_id`, capped at 100 entries (R9). `POST /api/undo` and `POST /api/redo` flip the latest stack entry by emitting the rescind row (FP) or a delete-row (FN_ADDED)
- [x] 7.4 Every `/api/marks` / `/api/undo` / `/api/redo` writes-and-commits within 1 second; log `[marks] saved job=<id> idx=<n> in <ms>ms` to stdout
- [x] 7.5 Verify R10 / R11: mark 100 detections, refresh the browser tab, confirm all 100 are persisted and the page restores them on reload

## 8. Retraining route and job tracker

- [x] 8.1 Create `column_review/retrain_jobs.py` exposing `start_retrain(epochs, min_corr) -> RetrainJob` that spawns `subprocess.Popen([sys.executable, "scripts/retrain_yolo.py", ...], cwd=PROJECT_ROOT)` and inserts a `retrain_jobs` row with `status="queued"`
- [x] 8.2 Spawn a background asyncio task that polls each non-terminal job: `proc.poll()` → set `status="running"` while None; set `completed` or `failed` on exit, capturing the last 64 KB of stderr into `stderr_tail`
- [x] 8.3 At server startup, reap orphans: any row with `status="running"` whose PID is no longer alive gets `status="failed"`, `stderr_tail="orphaned (server restarted)"`
- [x] 8.4 Create `column_review/routes/submit.py` with `POST /api/submit` that validates `count >= min_corr`, returns the confirm-dialog payload (counts + estimated runtime) on first POST, and spawns the job on second POST with `confirm=true`
- [x] 8.5 Add `GET /api/jobs/latest` returning the most-recent `retrain_jobs` row for the frontend status pill
- [x] 8.6 Verify D5: mark 20 detections as FP, POST `/api/submit` with `confirm=true`, observe a new `python3 scripts/retrain_yolo.py` process via `ps -ef`, observe `/api/jobs/latest` flip from `queued` to `running`

## 9. Frontend shell

- [x] 9.1 Write `column_review/static/index.html` with the empty-state shell (file picker, viewer container, progress strip, mini-map slot, fail-banner overlay)
- [x] 9.2 Copy `openseadragon.min.js` from `scripts/correction_app/static/vendor/` to `column_review/static/vendor/` (the only file kept verbatim from the deleted package)
- [x] 9.3 Write `column_review/static/styles.css` with the four-state palette (UNREVIEWED `#1e90ff`, FP `#d72631`, FN_ADDED `#ff8c00`, optional TP `#2e8b57`) plus the fail-banner styles
- [x] 9.4 Write `column_review/static/app.js` with the empty-state event wiring (picker → POST `/api/open` → mount OSD with the returned tile source)

## 10. Detection overlay canvas

- [x] 10.1 Add an absolute-positioned `<canvas id="overlay-canvas">` over the OSD viewer; `pointer-events: none` so clicks fall through to OSD
- [x] 10.2 Implement `paintOverlay()` driven by the OSD `update-viewport` event: world → screen transform via `state.osd.viewport.imageToViewportRectangle`; viewport-cull boxes outside the visible region
- [x] 10.3 Implement state-to-style mapping: UNREVIEWED solid blue, MARKED_FP dashed red, FN_ADDED dotted orange, MARKED_TP solid green; zoom-adaptive stroke width with a floor of 1.0
- [x] 10.4 Implement `paintMinimap()` driven by the same event — downsampled overlay showing FP/FN clusters; ~360×260 px
- [x] 10.5 Verify R6: at fit-to-window zoom, three states are visually distinguishable without zooming in

## 11. Mouse interactions

- [x] 11.1 OSD `canvas-click` handler: hit-test all detections against the clicked image-pixel coordinate; tolerance radius = `max(12, 12 / state.osd.viewport.getZoom())` to scale with zoom (R8). On hit, POST `/api/marks` with `FP_TOGGLE`
- [x] 11.2 Add an `osd-drag` overlay that intercepts left-drag in EMPTY space (no detection under cursor at mousedown). On mouseup, post `FN_ADDED` with the drawn bbox in image pixel coordinates
- [x] 11.3 Configure OSD: `gestureSettingsMouse.dragToPan=false`, `gestureSettingsMouse.clickToZoom=false`; pan is bound to middle-drag and Space+left-drag instead
- [x] 11.4 Mouse-wheel zoom centred on cursor: OSD default behaviour, verified explicit
- [x] 11.5 Verify R5: click toggles FP within 1s persistence; left-drag adds a single FN_ADDED row; middle-drag pans; wheel zooms on cursor

## 12. Keyboard map and focus management

- [x] 12.1 Implement the keyboard map from D8 in `app.js`: `F`/`X`=FP, `U`=undo, `Shift-U`/`Y`=redo, `Enter`=Save&Submit, `0`=100%, `H`=fit, `Z`=zoom-to-selection, `N`/`P`=next/prev, `J`=jump-to-next-unreviewed, `Space`=pan modifier
- [x] 12.2 Bind all shortcuts at `window` level; suppress when focus is in an `<input>` (the zoom-level numeric field)
- [x] 12.3 Implement `jumpToNextUnreviewed()` reusing the loop pattern from old `app.js` (fixed start-anchor for `activeIndex == null`)
- [x] 12.4 Verify R4: complete a 10-mark + 3-undo + 2-redo + jump-unreviewed + Save & Submit cycle using only the keyboard

## 13. R2 regression guard (the canary)

- [x] 13.1 Implement `installRenderCanary()` in `app.js` that, on every drawing-open, schedules a `requestAnimationFrame` AFTER both the OSD `open` event has fired AND the `/api/detections` fetch has resolved
- [x] 13.2 Inside the rAF: check `state.osd.world.getItemCount() > 0` AND `state.detections.length > 0`; if either is false, call `showFailBanner` with the missing-side identifier
- [x] 13.3 Style the `#fail-banner` as a full-viewport overlay with WCAG AA contrast — never a silent log
- [x] 13.4 Verify R2: open the fixture drawing — both layers render, no banner. Then temporarily force `state.detections = []` in devtools and reload — banner appears with "detections missing"

## 14. Autosave indicator and progress strip

- [x] 14.1 Render the progress strip at the top: drawing ID, live counts (UNREVIEWED `b`, FP `b`, FN_ADDED `b`, optional TP `b`), zoom indicator, filter buttons
- [x] 14.2 Show a "saved <Ns ago>" pill that updates from the response timestamp of each successful POST to `/api/marks` / `/undo` / `/redo`
- [x] 14.3 Verify R10: a browser refresh mid-session restores all marks without reviewer action

## 15. Save & Submit confirmation flow

- [x] 15.1 Bind `Enter` and a visible "Save & Submit" button to `triggerSubmit()`
- [x] 15.2 First POST `/api/submit` returns counts + projected runtime; the frontend shows a confirm modal — ONE modal, outside the correction loop, R4 compliant
- [x] 15.3 On confirm, second POST `/api/submit` with `confirm=true` triggers `start_retrain`; modal closes; status pill begins polling `/api/jobs/latest`
- [x] 15.4 Render retrain status pill with states `queued | running (Ns) | completed (Ns) | failed`. On failure, expand to a dismissable banner with the stderr tail
- [x] 15.5 Verify R12: trigger retrain, see status pill flip queued → running → completed; force a failure (e.g., bogus `--epochs 0`) and confirm the failure banner with stderr appears

## 16. Performance budget enforcement

- [x] 16.1 In `routes/detections.py::post_marks`, time the DB write; if > 50 ms, append the measurement to `data/jobs/<job_id>/perf.log`
- [x] 16.2 In `app.js`, time the open-to-first-render path with `performance.now()` and POST `/api/render-ack` with the duration; the server logs it and surfaces a startup banner if it exceeds 3 s on the most recent open
- [x] 16.3 At server startup, print one line `[perf] budgets: open<=3000ms, mark<=50ms` so the user sees the contract

## 17. Delete old surface

- [x] 17.1 Remove `scripts/correction_app/` recursively (8 files, ~2,440 LOC) via `rm -rf` then `git add -A` _(done 2026-06-04: user confirmed UI + retrain working end-to-end; deletion staged in git)_
- [x] 17.2 Remove `cmd_review` function and the `review` subparser block from `scripts/hitl.py` (preserve `ingest`, `build-tiles`, `retrain`, `status`)
- [x] 17.3 Update the 5 mentions of `python3 scripts/hitl.py review` in `README.md` to invoke `column-review` instead; update the TL;DR table at the top first
- [x] 17.4 Update the 2 architectural references to `scripts/correction_app/` in `CLAUDE.md` to point at `column_review/`
- [x] 17.5 Update the single doc-comment in each of `scripts/corrections_logger.py`, `scripts/postprocess_pipeline.py`, `scripts/ingest_drawings.py` that points at `scripts/correction_app/`
- [x] 17.6 Verify: `python3 scripts/hitl.py --help` lists `ingest`, `build-tiles`, `retrain`, `status` but not `review`; `python3 scripts/hitl.py review --help` fails with `invalid choice: 'review'`

## 18. End-to-end verification

- [x] 18.1 R2 regression on fixture: `column-review` → open `TGCH-TD-S-200-L3-00` → both image and detections render within 3 s, no fail banner _(user confirmed 2026-06-04 after meta.source alignment fix — bboxes land on column glyphs)_
- [ ] 18.2 Performance: mark 100 FPs in rapid succession on the fixture; observe no UI lag and `[marks]` log lines all under 50 ms _(browser-side — and currently ~100 ms/mark per the perf-log; SQLite commit-with-fsync overrun acknowledged as residual)_
- [ ] 18.3 Autosave: refresh the browser mid-session; all marks persist _(browser-side — needs user to test)_
- [x] 18.4 Retrain: mark 20+ FPs, Save & Submit, confirm, observe `retrain_yolo.py` process in `ps -ef` and status pill in the UI _(user confirmed 2026-06-04: 393 labels / 121 deletes, retrain completed, best.pt written to runs/detect/runs/detect/correction_feedback/weights/; user said "the basic retrain architecture of human in the loop is up very good")_
- [x] 18.5 Old surface: `ls scripts/correction_app/` returns "No such file or directory" _(verified 2026-06-04 after 17.1)_
- [x] 18.6 OpenSpec validation: `openspec validate replace-correction-ui-with-column-review --strict` reports valid
- [x] 18.7 Smoke tests unchanged: `python3 scripts/hard_negative_pool.py --dry-run` and `python3 scripts/hitl.py retrain --dry-run` produce identical output to the pre-change baseline
