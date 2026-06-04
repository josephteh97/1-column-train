## Context

The current correction-marking surface is `correct_detections.ipynb`, a 14-cell Jupyter notebook (cells 0–13: imports, knobs, inference, post-process, save-job, two-cell review-grid using `ipywidgets`, two-cell add-missed via a hand-edited `missed = [(cx, cy, size)]` list, and a print-cell that prints the retrain command). The notebook flow imposes a per-action full re-render, has no keyboard-first interaction model, no undo, no mini-map, no autosave-per-action, and loads the full A0/300DPI raster (~140 megapixels) as a single PIL bitmap into the kernel — none of which scale to thousands of marks per session. The user has decided the existing UI is unacceptable and must be deleted, not refactored.

The change is strictly scoped to the correction-marking surface. Everything around it stays:

- `scripts/corrections_logger.py` is the SQLite + filesystem storage layer (`new_job_id`, `save_job`, `record_delete`, `record_edit`, `record_add`, `iter_effective_corrections`, `summary`). It is imported by `scripts/retrain_yolo.py` and `scripts/hard_negative_pool.py`. The new web backend uses these functions directly.
- The `corrections` table schema (job_id, element_type, element_index, original_element JSON, changes JSON, is_delete, timestamp) and the `data/jobs/{job_id}/px_detections.json` shape (`columns: [{bbox, score, source?}]`) are the hard contracts the retrain pipeline reads. They MUST remain unchanged.
- `scripts/ingest_drawings.py::resolve_drawing(drawing_id) -> (raster_path, meta_dict)` is the existing resolution helper. The new FastAPI app calls it directly to locate raster + DZI per drawing.

Locked decisions from the user (via AskUserQuestion this session):

- Tech stack: web app — local FastAPI backend + static OpenSeadragon-based JS frontend.
- Tile pyramid: generated at ingestion time.
- Storage schema: frozen.
- Session scope: one drawing per session.

## Goals / Non-Goals

**Goals:**

- Delete `correct_detections.ipynb` in this change; the new web reviewer is the sole correction-marking surface from day one.
- Hit every one of the 12 hard UX requirements from the proposal (keyboard-first, tile-pyramid pan/zoom, navigation aids, four visual states, ≥100-level O(1) undo/redo, zoom-adaptive hit-test, single-drag FN add, batch ops, progress UI + jump-to-next, autosave per action, <50 ms interaction lag with load-time hard-fail).
- Preserve the corrections-DB write contract so that `scripts/retrain_yolo.py` and `scripts/hard_negative_pool.py` continue to work without any code change.
- Add ingestion-time DZI tile-pyramid generation with a backfill subcommand for drawings ingested before this change.
- Refuse to silently degrade: missing DZI → loud error; perf-budget miss at load → loud error.

**Non-Goals:**

- Detection model, training pipeline, inference, post-processing, retrain, hard-negative pool — untouched.
- Multi-reviewer collaboration, audit logs, comment threads on detections, OCR / text extraction.
- Any other UI surface (inference dashboards, training monitors, ingestion-CLI UX redesign).
- Migration tooling for old correction data — schema is frozen so no migration is needed.
- Coexistence of old and new UIs. No flag-gated rollout, no A/B test, no transition period.
- Authentication or multi-user support. Reviewer-id is a single string prompted once and persisted to `~/.column-review.json`.

## Decisions

### Web app (FastAPI + static OpenSeadragon) over PySide6 + QGraphicsView

OpenSeadragon is a battle-tested deep-zoom viewer that handles DZI natively, with built-in pan/zoom inertia, cursor-anchored wheel zoom, and HiDPI awareness. Implementing the same on QGraphicsView would require writing tile-pyramid management, LRU cache, and cursor-anchored zoom from scratch. The web tech stack also avoids platform-specific install pain (Qt wheel sizes, GL drivers) and runs identically on Linux, macOS, and Windows. The single-user local-host loopback bind (default 127.0.0.1:8765) sidesteps every concern that argues against web tech for a desktop tool. Alternatives considered: PySide6 + QGraphicsView (rejected for the above reasons), Tauri/Electron (extra install ceremony with no benefit over plain browser), raw native canvas (reinvents OSD).

### DZI tile pyramid generated at ingestion time, not on first open

Generating the pyramid lazily on first open would push 10–30 s of work into the reviewer's interactive critical path on every fresh drawing, violating the <3 s open requirement. Generating at ingestion time amortises the work into the batch ingest step (where the reviewer is not waiting interactively) at the cost of ~25–35% extra disk per drawing. A0/300DPI tile generation with Pillow is ~10–30 s on a single core; we accept the time at ingest because the reviewer's interactive critical path is the precious budget. A backfill subcommand `python3 scripts/hitl.py build-tiles <id>` covers drawings ingested before this change. The `--no-tiles` opt-out exists for ingest-only-no-review use cases, but the web reviewer refuses to open such drawings with a loud diagnostic — no silent single-bitmap fallback.

### Custom canvas overlay layer, not OSD's DOM-overlay layer

OpenSeadragon's standard `Overlay` mechanism creates one DOM node per overlay. At 2000 detection boxes this pushes the layout thread well past the 50 ms budget on a single mark (style invalidation + reflow cascades across 2000 elements). The chosen approach: a single full-viewport canvas pinned to OSD's viewport via the `update-viewport` event, with world-to-screen mapping done in our code and bbox culling outside the visible viewport. On a mark, only the changed box's overlay region and the progress counters are redrawn; the underlying DZI tile layer is untouched. This is the mechanism that makes the <50 ms interaction-lag requirement achievable.

### TP/FP/FN_ADDED mapping onto the frozen schema with additive sidecar tables

The existing `corrections` table only knows about deletions and edits (the FP and FN cases), because the original notebook UI did not record TP confirmations — a TP in the old flow was the implicit "no correction recorded" state. The new UI MUST record TP marks per the autosave-per-action requirement (otherwise undoing a TP mark or restoring state after a crash is impossible).

Mapping choices:

- FP → row in existing `corrections` table with `is_delete=1`. No change.
- FN_ADDED → append to `data/jobs/{job_id}/px_detections.json` under `"columns"` with `source: "human_added"`, exactly as the prior notebook did via `record_add`.
- TP → new sidecar table `tp_confirmations(session_id TEXT, job_id TEXT, element_index INTEGER, ts REAL, PRIMARY KEY (job_id, element_index))`.

This is additive only: no existing column, index, or constraint is touched. `scripts/retrain_yolo.py` and `scripts/hard_negative_pool.py` continue to read the `corrections` table and call `iter_effective_corrections` unchanged. They simply ignore the new sidecar tables. We treat "schema frozen" as "existing tables frozen; sidecar additions for UI-only state are allowed". The strict-frozen alternative (do not record TP at all, accept that TP marks evaporate on reload) is rejected because requirement 11 (autosave per action) plus requirement 8 (undo of any action including TP) jointly require persistence. The user can override during /opsx:apply if they prefer the strict reading.

Reviewer identity gets a second sidecar `reviewer_sessions(session_id TEXT PRIMARY KEY, reviewer_id TEXT NOT NULL, started_ts REAL NOT NULL)`. The reviewer_id is established by a single first-launch input persisted to `~/.column-review.json`; subsequent launches inject a fresh `session_id` and write the row. `tp_confirmations.session_id` is a foreign-key-by-convention reference into `reviewer_sessions`.

### Undo/redo: in-memory ring buffer of inverse-ops, O(1) per action

A 100-deep ring buffer of `{action_id, inverse_op}` records is allocated on session start. Each marking action pushes its inverse onto the buffer (e.g. mark-TP's inverse is "remove the `tp_confirmations` row plus revert the in-memory state to unreviewed"). Push, pop, and re-apply are constant-time pointer increments + one SQL DELETE or INSERT. Beyond 100 levels, the oldest inverse is dropped from the buffer; the action is no longer reversible but the persisted DB state is unchanged. The buffer is in-memory only (closing the tab loses the undo history but not the persisted marks).

### Zoom-adaptive hit-test radius via screen-space tolerance

OSD exposes the current zoom factor `viewer.viewport.getZoom()`. The hit-test radius is computed at click time as `max(base_tolerance_css_px, base_tolerance_css_px / zoom_factor)` — at high zoom the tolerance shrinks toward `base_tolerance_css_px` so big-on-screen boxes do not catch stray clicks; at low zoom the tolerance grows so small-on-screen boxes remain clickable. The `base_tolerance_css_px` default is 8; configurable via `--hit-tolerance-px`.

### Autosave per action: synchronous SQLite write on the request handler thread

FastAPI POST handlers for marking actions write synchronously and return only after the SQLite transaction commits and the `px_detections.json` file is fsync'd via `os.replace`. This guarantees the 1-second-durability requirement without a separate flush worker. The frontend awaits the response before showing the mark as "saved" (gated on a boolean return from `applyMark`); if the response fails, the undo stack is NOT pushed and a loud inline error appears. No optimistic UI for marking actions: durability beats latency, and the per-action server round-trip is well under the 50 ms budget on loopback. Crash-safety ordering inside `_apply_marks`: write the JSON file FIRST via `os.replace`, then commit the SQLite transaction. A kill -9 between the two leaves an FN_ADDED visible to downstream consumers (the JSON entry has `source:human_added`) without a corrections row — strictly safer than the reverse, which would have left an orphaned DB row pointing at a JSON index that never existed.

### Single-writer storage policy (`_apply_marks` inlines the SQL)

The original design called for the FastAPI backend to import `record_delete`, `record_edit`, `record_add`, and `save_job` from `scripts/corrections_logger.py`. Implementation found three reasons this had to flip:

1. **Batch-as-one-transaction**: each `record_*` helper opens, commits, and closes its own SQLite connection. Routing a 500-mark rubber-band through them would issue 500 fsync'd commits, contradicting requirement 5's "applying each in one transaction" promise. Inlining the SQL into `_apply_marks` lets one connection wrap the whole batch.

2. **DELETE_FN rescind trap**: `record_delete` inserts an `is_delete=1` row. `iter_effective_corrections` then RESCINDS it because the FN_ADDED's existing `is_delete=0` row is still present. To honour DELETE_FN correctly the writer must DELETE the prior `is_delete=0` row first AND tag the new `is_delete=1` row with `changes={"action":"delete_fn"}` so the state map can distinguish "user undid their own add" (REMOVED → hidden) from "user marked the human_added entry as FP" (visible FP). The legacy helpers had no path to express either.

3. **Bootstrap latency**: `save_job` JPEG-encodes the full A0/300DPI raster (`quality=92`, `optimize=True`) inside `create_app`, blocking 10–30 s on first launch. `_bootstrap_empty_job` writes only the lightweight `px_detections.json` and `_spawn_render_jpg_write` runs the encode on a daemon thread so the reviewer's first-paint stays under the 3-second budget.

Consequence for the corrections_logger module: `save_job`, `record_delete`, `record_edit`, `record_add`, the `JobAlreadyCorrected` exception, and the private `_load_px_detections` / `_write_px_detections` helpers are REMOVED — they had zero live callers after the notebook deletion. The live public surface is `new_job_id`, `iter_effective_corrections`, `summary`, plus the path constants and `_SCHEMA` DDL. The on-disk schema (the actual storage contract that downstream `scripts/retrain_yolo.py` and `scripts/hard_negative_pool.py` consume) is unchanged — what's gone is the Python convenience layer that nothing now imports.

### Performance probe at load time

On page load the frontend runs a one-shot probe:

1. Verify DZI exists (HEAD request to `<id>.dzi`). Missing → loud failure naming the build-tiles command.
2. Allocate a synthetic 2000-box overlay test set and render it once. Measure the time. > 50 ms → loud failure naming the bottleneck.
3. Measure tile fetch latency for level-0 and the current viewport-level tiles. If level-0 cannot render within the first 3 s → loud failure naming "DZI fetch latency".

The probe runs before any reviewer interaction is enabled. There is no fallback path that silently lowers fidelity.

### On-demand inference via `POST /api/infer`

The original design called for inference to run upstream (`test_column.ipynb`) and for the reviewer to consume a pre-populated `px_detections.json`. Two reviewer sessions on the user's TGCH plan made this contract unworkable in practice: the user saw "grey blank nothing" because no detections were populated, and the spec's pointer at the notebook turned the web reviewer into a two-tool workflow.

The amendment: the reviewer exposes a `POST /api/infer` endpoint and a single green "Run inference" button in the progress strip. The endpoint runs `column_detect.pt` against the canonical raster via `scripts/tiled_inference.py::tiled_predict`, post-processes through `scripts/postprocess_pipeline.py::run_pipeline` (the 6-filter pipeline + OOD hard-fail), and writes the resulting columns atomically via `os.replace`. Synchronous on the request-handler thread (FastAPI's sync-`def` threadpool); inference time is ~2–5 s on GPU and ~30–90 s on CPU. The frontend disables the button + shows a spinner while a POST is in flight; the terminal emits a per-tile progress trace (one line every ~N/10 tiles) so the user can verify the call isn't stuck.

Key contract clauses:

- **Reviewer-id required** — `_require_session()` gate; 409 until the prompt is submitted.
- **Refusal to overwrite** — 409 if `px_detections.json` already has any columns (model-produced OR `source: "human_added"`). The button is hidden when detections exist, so the gate is defence-in-depth against curl / DevTools bypass.
- **Loud failures, not opaque 500s** — missing raster → 404 with a re-ingest hint; missing weights → 500 naming the path; tiled_predict / run_pipeline exceptions → 500 with the stage name; OOD hard-fail in run_pipeline lands in the same loud-500 path.
- **Weight-load memoisation** — the `column_detect.pt` model is cached in module scope keyed on file mtime, so iterating tuning parameters (`--conf-th 0.20` vs `0.25`) doesn't re-read the 57 MB weights file every click. Promoting a fine-tuned weight via `cp column_detect_ft_<ts>.pt column_detect.pt` invalidates the cache on the next request.

The endpoint is documented as a normative `### Requirement: On-demand inference via Run-inference button` in `correction-ui/spec.md` so the contract is the source of truth, not the implementation.

## Risks / Trade-offs

- **[Frozen-schema strictness]** The user's locked decision says "frozen". Sidecar tables are additive but a strict reading bars even adding tables. → Mitigation: the design names the sidecar tables explicitly and the user can veto during /opsx:apply; the alternative (TP not persisted) is documented above as the strict-frozen fallback.
- **[DZI generation slows ingest]** A0/300DPI tile generation costs ~10–30 s per drawing. → Mitigation: documented in `--help` and `READMD.md`; `--no-tiles` opt-out exists; the cost is one-time per drawing and lives outside the reviewer's interactive critical path.
- **[OSD overlay perf cliff]** Standard OSD DOM-overlay with 2000 nodes blows the 50 ms budget. → Mitigation: custom canvas overlay layer (see Decisions); the load-time perf probe surfaces any unexpected regression loudly.
- **[Reviewer_id has no auth]** Single-user assumption; any reviewer can claim any id. → Acceptable for the single-user local-host loopback use case; out of scope per non-goals.
- **[Browser dependency]** Requires a modern browser. → Acceptable; every developer machine in the user's flow has one.
- **[Schema typo risk]** A typo in the spec's verbatim column names silently misaligns with the actual table. → Mitigation: the smoke-test task in tasks.md group G5 runs `retrain_yolo.py --dry-run` and `hard_negative_pool.py --dry-run` against a session's writes, which would surface drift immediately.
- **[Disk cost]** ~25–35 % extra disk per drawing for DZI tiles. → Documented; the alternative (lazy generation) is worse for UX.
- **[Loopback bind]** Default 127.0.0.1 prevents remote access. → Intentional; remote review is a non-goal.

## Migration Plan

This change has no data migration (existing `corrections` rows and `px_detections.json` files are untouched). The deployment sequence:

1. /opsx:apply implements the change: writes the FastAPI app + static frontend, extends `ingest_drawings.py` with DZI, adds `review` and `build-tiles` subcommands to `hitl.py`, deletes `correct_detections.ipynb`, updates `READMD.md` and `CLAUDE.md`.
2. On first launch the FastAPI app runs the additive sidecar-table migration (idempotent `CREATE TABLE IF NOT EXISTS`) against `data/corrections.db`.
3. For drawings ingested before this change, the reviewer runs `python3 scripts/hitl.py build-tiles <id>` once per drawing to generate the DZI backfill.
4. The 172 deletes already recorded in `data/corrections.db` survive verbatim — `retrain_yolo.py --dry-run` and `hard_negative_pool.py --dry-run` are run as part of /opsx:apply G5 smoke-tests to confirm.

Rollback: revert the /opsx:apply commit. The deleted notebook is recoverable from git history. The sidecar tables remain in `data/corrections.db` but are ignored by all other code; they can be dropped manually with two `DROP TABLE` statements if desired.

## Open Questions

- **Reviewer-id source**: the design proposes a first-launch prompt persisted to `~/.column-review.json`. An alternative is to read `$USER` or `git config user.name`. The user can choose at /opsx:apply.
- **Snap-to-grid spacing**: requirement specifies optional snap-to-grid for FN add but does not commit a grid size. Default proposed: 8 pixels at 1:1 zoom, configurable via `--snap-grid-px`.
- **Tile JPEG quality**: default 80. If the user finds the tile compression visible at high zoom, raise to 90 (~+15% disk cost).
- **Server port**: default 127.0.0.1:8765. If 8765 is in use, the launcher picks the next free port and prints it. No persistent config file for port choice in this change.
