## Context

### Current state

The repo has a FastAPI + OpenSeadragon correction reviewer at
`scripts/correction_app/` (8 files, ~2,440 LOC):

| File | LOC | Role |
|---|---|---|
| `app.py` | 1,177 | FastAPI backend, `POST /api/infer`, mark-apply, state-map |
| `static/app.js` | 1,040 | Frontend (OpenSeadragon mount, overlay canvas, mark/state UI) |
| `static/index.html` | 64 | Single-page shell |
| `static/styles.css` | 142 | Theme |
| `static/vendor/openseadragon.min.js` | ~8 | Vendored OSD library |
| `__init__.py` | 9 | Package marker |

It is launched via `python3 scripts/hitl.py review <drawing-id>` (handler
`cmd_review` at `scripts/hitl.py:141-211`, subparser at `:300-338`).

The reviewer's lived experience: opening a floor plan renders blank. Image
and detections never both render together. The correction loop is dead
end-to-end. Multiple patch passes (CUDA auto-detect, UX redesign, 5-bug
fix batch) have NOT closed the underlying "blank canvas" failure mode.

### Surrounding infrastructure (kept)

| Module | Purpose | Status |
|---|---|---|
| `scripts/corrections_logger.py` | SQLite layer for `data/corrections.db` (`corrections` + helpers). Public: `new_job_id`, `iter_effective_corrections`, `summary`. | UNCHANGED |
| `scripts/retrain_yolo.py` | Fine-tunes `column_detect.pt` from `data/corrections.db` rows. CLI: `--epochs`, `--min-corrections`, `--imgsz`, `--base-weights`, `--dry-run`. | UNCHANGED |
| `scripts/ingest_drawings.py` | Builds the DZI tile pyramid at `data/raw/drawings/<id>.dzi` + `<id>_files/` (256-px tiles, 1-px overlap, JPEG q=80). | UNCHANGED |
| `scripts/postprocess_pipeline.py` | YOLO post-processing (NMS, tile-boundary merging). | UNCHANGED |
| `scripts/hitl.py {ingest, build-tiles, retrain, status}` | The other four subcommands of the same CLI. | UNCHANGED |
| `column_detect.pt` (repo root) | Deployed YOLO weights. | UNCHANGED |
| `data/corrections.db` SQLite schema | `corrections`, `tp_confirmations`, `reviewer_sessions`. | UNCHANGED (sidecar `CREATE TABLE` ownership moves) |

### Constraints

- Single reviewer per session (no auth, no multi-tenancy).
- Single-machine deployment (ThinkStation P340, RTX 4000 8 GB, modern Chrome / Firefox).
- No CI in this repo; verification is local + manual browser.
- Reviewer comfort over thousands of marks/session is the dominant UX axis.
- A0 @ 300 DPI ≈ 9933×14043 px ≈ 140 MP — tile pyramid is non-negotiable.

## Goals / Non-Goals

**Goals**

- Deliver a single-command `column-review` that launches a working
  correction loop from any directory after `pip install -e .`.
- Image + detections always render together; the blank-canvas failure
  mode of the old page is detected and surfaced as a fail banner, never
  silent.
- A0 drawings open within 3 s; 60 fps pan/zoom; ≤50 ms interaction lag
  on 2 000 detections.
- Keyboard-first correction loop (FP, FN-add, undo, redo, save) with
  no modals inside the loop.
- Save & Submit fires `scripts/retrain_yolo.py` as a background
  subprocess after a single confirm dialog; the UI never blocks on
  retrain; failure is visible with stderr tail.
- Delete the old `scripts/correction_app/` package and the `hitl.py
  review` subcommand in the same change. All callers and user-facing
  docs updated in the same change.

**Non-Goals**

- Multi-reviewer collaboration, audit log, comment threads.
- OCR or text extraction from drawings.
- Cloud deployment, authentication, multi-tenancy.
- Migration tooling for legacy correction data (the old page never
  produced any).
- Surfaces outside the correction loop (inference dashboards, training
  monitors, ingestion UI).
- Archiving the prior `rebuild-correction-ui-web` change directory
  (separate `/opsx:archive` follow-up).
- Inset-active-ring / animated auto-pan polish (deferred from prior
  iteration).

## Decisions

### D1. Greenfield `column_review/` package with hybrid DB layer

**Decision.** Ship a new top-level Python package `column_review/` that
owns the CLI, server, frontend, and inference plumbing. It IMPORTS
`scripts/corrections_logger.py` for the canonical SQLite layer rather
than reimplementing it. The two sidecar tables (`tp_confirmations`,
`reviewer_sessions`) whose `CREATE TABLE` statements live today in
`scripts/correction_app/app.py` move INTO `column_review/db.py` and are
issued at startup.

**Why not rewrite in-place under `scripts/correction_app/`?** That path
requires the launch command to remain `python3 scripts/hitl.py review
<drawing-id>` (a multi-token invocation from inside the project root)
and contradicts R1's "single short shell command from any directory
after `pip install`". A top-level package with a `[project.scripts]`
console_script meets R1 cleanly and gives the new UI a fresh import
surface uncoupled from `scripts/`.

**Why not full greenfield with a brand-new DB layer?** Re-implementing
`corrections_logger.py` doubles maintenance and risks drift from
`scripts/retrain_yolo.py`'s read contract. Sharing the proven module
collapses the contract surface to one file.

**Alternative considered: separate Python service for inference.** Spawn
inference as its own daemon and have the UI hit it over HTTP. Rejected:
adds an IPC failure mode for no benefit on a single-box deployment.

### D2. OpenSeadragon over a pre-built DZI pyramid

**Decision.** Continue to use OpenSeadragon as the deep-zoom renderer
(R3). Tile pyramids are pre-built on disk by `scripts/hitl.py ingest`
(unchanged). `column-review` READS, never writes, the pyramid.

**Why OSD over Leaflet / Mapbox?** OSD is already vendored in the
repo and known to work for this dataset; the failure mode that
motivated this change is in the page glue, not OSD itself.

**Why not generate the pyramid on first open?** Pyramid generation on
A0 takes 10–30 s — would blow R3's 3 s open budget. Pre-built is the
only path that hits the budget.

**Missing pyramid handling.** If the user picks a drawing whose
`<id>_files/` directory doesn't exist, the UI surfaces a typed error
naming the missing path and the exact `hitl.py ingest` invocation
needed. No stalling, no silent failure.

### D3. Frontend: vanilla JS, no build step

**Decision.** Single-page vanilla JS frontend served by FastAPI's
`StaticFiles`. No bundler, no React/Vue, no build step at run time.

**Why?** R1 forbids a runtime build step. The old UI's JS (1,040 LOC)
was vanilla too and the failure was not framework-related. A
build-less stack also keeps `column-review` debuggable without
node/npm on the deployment box.

**Trade-off accepted.** No type checking on the frontend. Mitigation:
the R2 regression guard (D7) makes the most damaging failure mode
loudly visible at runtime.

### D4. SQLite-backed corrections, with sidecar tables for FN_ADDED and TP

**Decision.** Keep `data/corrections.db` exactly as today:

```sql
corrections (
  id INTEGER PRIMARY KEY, job_id TEXT, element_type TEXT,
  element_index INTEGER, original_element TEXT, changes TEXT,
  is_delete INTEGER DEFAULT 0, timestamp REAL DEFAULT (strftime('%s','now')))
-- UNIQUE INDEX (job_id, element_index, is_delete)

tp_confirmations (session_id TEXT, job_id TEXT, element_index INTEGER, ts REAL,
  PRIMARY KEY (job_id, element_index))

reviewer_sessions (session_id TEXT PRIMARY KEY,
  reviewer_id TEXT NOT NULL, started_ts REAL NOT NULL)
```

Encoding:

| State | Where written | How |
|---|---|---|
| `MARKED_FP` | `corrections` | row with `is_delete=1`, `element_index = source index in px_detections.json` |
| FP undo | `corrections` | row with `is_delete=0` for the same `(job_id, element_index)`; the rescind invariant in `iter_effective_corrections` already handles this |
| `FN_ADDED` | `px_detections.json` + `corrections` | append `{"bbox": [...], "source": "human_added"}` to the JSON columns list (atomic temp-file replace), then write a `corrections` row with `is_delete=0` keyed by the newly-assigned positive `element_index`. This is the encoding `scripts/retrain_yolo.py` already consumes — it reads bboxes from the JSON and uses corrections as a delete-set |
| `MARKED_TP` (optional) | `tp_confirmations` | one row per accepted detection; absence ⇒ implicitly accepted |

**Why append to the JSON?** `scripts/retrain_yolo.py:432` reads bboxes
solely from `px_detections.json["columns"]` and uses correction rows
only as a delete-set keyed by `(element_type, element_index)`. Any
FN_ADDED encoding that lives only in the DB (e.g., negative-index
rows) would be silently dropped by the downstream retrain consumer.
The append-to-JSON encoding is the old correction_app's encoding too
— retaining it keeps the data-contract identical and avoids parallel
changes to `retrain_yolo.py`.

**Why no schema migration?** `scripts/retrain_yolo.py` already reads
this schema in production. Changing the shape would force a parallel
update there and risk silent breakage in the only downstream consumer.

### D5. Save & Submit fires retrain via `subprocess.Popen`

**Decision.** `POST /api/submit`, on Confirm, runs:

```python
subprocess.Popen(
    [sys.executable, "scripts/retrain_yolo.py",
     "--epochs", str(epochs), "--min-corrections", str(min_corr)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    cwd=PROJECT_ROOT,
)
```

The launched PID, start time, and a streaming tail of stderr are
persisted in a new lightweight SQLite table `retrain_jobs (id INTEGER
PK, pid INTEGER, started_ts REAL, status TEXT, finished_ts REAL,
stderr_tail TEXT)` inside the same `data/corrections.db`. The UI polls
`GET /api/jobs/latest` every 2 s while a job is non-terminal and
flips its status pill: `queued → running → completed | failed`.

**Why subprocess over import-and-call?** Retraining mutates global
ultralytics state and takes minutes. Embedding it in the FastAPI
process risks request-thread blocking, memory leakage, and forces an
ugly process lifecycle. `Popen` is the same shape `hitl.py retrain`
already uses; consistency wins.

**Why not a job queue (celery, RQ)?** Single-box, single-user
workload. Queue infrastructure is overkill and introduces operational
complexity (broker, worker process, supervisor) for no functional gain.

**Confirm dialog.** A one-step modal is shown OUTSIDE the correction
loop (after the reviewer has clicked Save & Submit) with the
correction counts and the runtime estimate. This does not violate the
"no modals in the loop" rule.

### D6. Move `_get_or_load_model`, `tiled_predict`, `pick_port`, `open_browser_soon` into the new package

**Decision.** The four helpers in the deleted `app.py` that are not
FastAPI-coupled are extracted into:

- `column_review/inference.py` ← `_get_or_load_model`, the `tiled_predict`
  wrapper, and the `POST /api/infer` handler body (rewritten as an
  async route handler that imports `column_review.inference` for the
  heavy lifting).
- `column_review/server.py` ← `pick_port(start)` and
  `open_browser_soon(url, delay_seconds)`.

The model weights cache (`(path, mtime, size)` key, `threading.Lock`
for concurrent first-load serialisation) survives unchanged from the
old code.

### D7. R2 regression guard (the canary)

**Decision.** A frontend invariant check after every drawing-open
cycle:

```js
window.requestAnimationFrame(() => {
  const hasImage = !!state.osd?.world.getItemCount();
  const hasDetections = state.detections?.length > 0;
  if (!hasImage || !hasDetections) {
    showFailBanner(
      `Renderer state inconsistent: ${hasImage ? "" : "image "}` +
      `${hasDetections ? "" : "detections "}missing`);
  }
});
```

The banner element is a full-viewport overlay (`#fail-banner` in
`styles.css`) with WCAG AA contrast on the failure colour, NOT a
silent log. This is the single hardest acceptance test that ties
back to the bug that motivated the change.

**Why this is sufficient.** The blank-canvas failure mode of the old
page corresponds exactly to one or both of `hasImage` /
`hasDetections` being false after open. If a future regression
re-introduces the failure, the user sees it; QA does not depend on a
human noticing "the page is blank".

### D8. Keyboard map

| Key | Action |
|---|---|
| `F` or `X` | Mark hovered/active detection FP |
| `U` | Undo |
| `Shift-U` or `Y` | Redo |
| `Enter` | Save & Submit (opens confirm dialog) |
| `0` | 100% zoom |
| `H` | Fit-to-window (home) |
| `Z` | Zoom-to-selection (active detection) |
| `N` | Next detection |
| `P` | Previous detection |
| `J` | Jump-to-next-unreviewed |
| `Space` (held) | Pan modifier for left-drag |

Reused from the old app where the binding was sensible; conflicts
resolved in favour of the proposal-prompt's nomenclature.

### D9. Performance budget enforcement

**Decision.** At server startup AND at every drawing-open, `column_review`
measures:

- Open-to-first-render: `time.perf_counter` from `POST /api/open` to
  the client's `osd.world.addItem`-complete event reported back over
  `POST /api/render-ack`.
- Mark-to-persist: server-side timer in `routes/detections.py::post_marks`.

If either exceeds its budget (3 s open, 50 ms mark) on a representative
fixture, a startup banner reports it in the UI with the measured value
and a plausible bottleneck name. Per R11, "fail loudly, don't degrade
silently".

## Risks / Trade-offs

- **[Risk] Tile pyramid missing for a drawing user wants to review.**
  Old flow auto-built one in `hitl.py ingest`; new flow assumes it
  exists. → Mitigation: file-picker surfaces a typed error with the
  exact `hitl.py ingest` command. Documented in `correction-ui` spec.

- **[Risk] Subprocess lifecycle.** If the user kills the `column-review`
  server mid-retrain, the subprocess survives (good), but the
  `retrain_jobs` table will never see a status flip and will report
  the job as "running" forever. → Mitigation: at server startup,
  `column_review/retrain_jobs.py` walks any `status='running'` rows
  and checks `os.kill(pid, 0)` — if the PID is gone, mark `failed`
  with `stderr_tail="orphaned (server restarted)"`.

- **[Risk] Tile cache 512 MB on an A0 drawing at 300 DPI.** With OSD's
  default tile cache, navigating wide regions can exceed the budget
  faster than LRU can evict if pan velocity is high. → Mitigation:
  configure OSD with `maxImageCacheCount` derived from the budget /
  tile bytes ratio; cap pan velocity client-side via OSD's
  `springStiffness` if profiling shows the cap is being exceeded.

- **[Risk] Vanilla JS loses to ad-hoc state management at ~2 000
  detections.** No reactive framework means manual cache invalidation
  for the overlay. → Mitigation: the overlay only repaints on three
  events (zoom/pan changed → `update-viewport`; a single detection
  mark changed → that one box only; bulk reload → full repaint). The
  paint path uses viewport culling and is already battle-tested in
  the prior repaint loop.

- **[Risk] Reviewer accidentally clicks Save & Submit.** Retrains take
  minutes on the RTX 4000. → Mitigation: confirm dialog (D5) names
  the correction counts and the projected runtime; the dialog is the
  only modal in the workflow.

- **[Risk] Frontend canary (D7) fires spuriously.** A slow tile-pyramid
  HTTP response could make `hasImage` false at the moment of the rAF
  check. → Mitigation: gate the canary to fire only AFTER OSD's
  `open` event fires (which is OSD's "first source is ready"
  guarantee) AND the `/api/detections` fetch has resolved (not just
  been initiated).

- **[Trade-off] `column-review` is a single-machine tool.** It will
  never grow into a multi-reviewer service. This is the explicit
  scope. If that requirement ever surfaces, a `column-review-server`
  daemon with auth is a separate change, not an evolution of this
  one.

## Migration Plan

This is a same-change replacement, not a phased rollout.

1. `column_review/` package + `pyproject.toml` entry are added.
2. Server, routes, frontend, retrain trigger, and R2 canary are built.
3. `scripts/correction_app/` directory is deleted.
4. `scripts/hitl.py review` subcommand is removed.
5. `README.md`, `CLAUDE.md`, and doc comments in
   `scripts/corrections_logger.py`,
   `scripts/postprocess_pipeline.py`,
   `scripts/ingest_drawings.py` are rewritten to point at the new
   command.
6. After merge, the user runs `pip install -e .` once. From that
   point, `column-review` is the only correction surface.

**Rollback.** `git revert <merge-sha>` brings back the
`scripts/correction_app/` directory and the `hitl.py review`
subcommand. The DB schema is unchanged in both directions, so no DB
migration is required to roll back. No state in `data/corrections.db`
is invalidated by either direction.

## Open Questions

None at design time — the four user decisions (hybrid replacement,
keep DB schema, confirm dialog before retrain, local subprocess
retrain) close the open questions enumerated in the original
proposal prompt. Discovery during apply may surface implementation
details; those are handled in `/opsx:apply` per the OpenSpec fluid-
workflow model rather than blocking design.
