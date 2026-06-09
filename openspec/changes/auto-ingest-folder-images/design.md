## Context

`column-review` ships with two ways to open a drawing:

1. **DZI path** (`POST /api/open` → OSD DZI tileSource). Fast,
   downstream-friendly, but requires a prior CLI ingest step
   (`python3 scripts/hitl.py ingest <plan> --drawing-id <id>`) that
   writes `data/raw/drawings/<id>.{png,jpg}`, `<id>.meta.json`,
   `<id>.dzi`, and `<id>_files/`. Without that, the endpoint hard-fails
   with HTTP 412.
2. **Image-mode path** (`POST /api/open-local-image` → OSD image
   tileSource). Skips the DZI step, serves the raw raster directly.
   Acceptable on small plans, slow on A0-class images, and any future
   tooling that assumes a real DZI on disk (the recently-added
   annotated-image export at full resolution is the canonical example)
   silently breaks for image-mode opens.

The user has consolidated every retrain image into
`~/Documents/retrain-dataset/` (71 files today) and wants the picker
to behave as a single-click "drop image → start reviewing" entry
point. The CLI ingest hop is the last manual step on that path.

## Goals / Non-Goals

**Goals:**
- One click from picker to OSD-with-DZI for any file in
  `~/Documents/retrain-dataset/`, with no separate CLI step.
- Idempotent re-clicks: the second click on an already-ingested file
  is instant and does not rebuild the DZI.
- Same `column-review` launch command. No new flags, no rename.
- Hard-wire the watched folder to `~/Documents/retrain-dataset/` so
  the picker cannot accidentally surface files from anywhere else.

**Non-Goals:**
- Background-job pattern for ingest. Synchronous in v1; escalate to
  the `retrain_jobs` pattern only if wall time becomes a UX problem.
- Folder watching (inotify/FS events). User refreshes the picker.
- Configurable DPI from the UI. Package default
  (`ingest_drawings.INPUT_DPI`, currently 300) only.
- Removing `/api/open` (DZI-only). Keep the path; this change just
  removes the reason a user would ever need to call it first via CLI.

## Decisions

### D1 — Reuse `POST /api/open-local-image` (modify in place)

Two endpoint shapes were considered:

- **A.** Add a new `POST /api/ingest-and-open` and leave
  `/api/open-local-image` unchanged (kept as image-mode escape hatch).
- **B.** Modify `/api/open-local-image` so its response always points
  at a freshly-built (or pre-existing) DZI. Drop the
  `tile_source_type: "image"` response key.

Chosen: **B**. The image-mode path has no remaining caller after the
folder is hard-wired to a single curated location, and split endpoints
double the surface area for picker behavior we want to keep coherent.
Keeping the same request shape (`{filename, reviewer_id}`) means the
frontend change is one line.

### D2 — Synchronous ingest, frontend spinner

A0 PDF rasterise + DZI build is ~30–60 s on this machine. Three
options:

- **A.** Run synchronously, frontend spinner on the picker button.
- **B.** Spawn into the existing `retrain_jobs` background-job
  table, poll status.
- **C.** Inline `BackgroundTasks` in FastAPI plus a separate `GET
  /api/ingest-status/<drawing_id>` poll.

Chosen: **A**. The `retrain_jobs` machinery exists for >5-minute,
GPU-bound work where the user wants to keep the UI usable. A
single-shot 30–60 s ingest with the user already paused on the
"open this drawing" intent is a fair fit for a button spinner. If
wall time pushes past 60 s in practice, escalating to B is a
mechanical follow-up — the endpoint contract on the response side
doesn't change.

### D3 — Idempotency via `resolve_drawing(drawing_id)`

`scripts.ingest_drawings.resolve_drawing(...)` already returns
`(raster_path, meta)` with a `dzi_path` field populated when the DZI
manifest exists on disk. On entry to the handler:

1. Derive `drawing_id = raster_path.stem` (current behaviour).
2. Call `resolve_drawing(drawing_id)`; on `FileNotFoundError`,
   proceed to the ingest path.
3. If `meta["dzi_path"]` is set AND the file still exists on disk,
   short-circuit: skip the rebuild, just bootstrap the job + session
   and return.

This makes re-clicks on a known drawing essentially free and
preserves corrections already associated with the stem-based id.

### D4 — Hard-wire `images_dir`, remove `--images-dir`

Two options for the folder source:

- **A.** Keep `--images-dir` as an override on top of a hard-coded
  default.
- **B.** Drop the flag; hard-wire
  `Path("~/Documents/retrain-dataset").expanduser()` as the only
  value `column_review/cli.py` ever puts into `config["images_dir"]`.

Chosen: **B**. The user has been explicit ("only take from this
folder"). Keeping the flag invites a future "wait, why are my files
not showing?" debugging path. Missing-directory fallback stays the
same as today: log one warning line and continue with `images_dir =
None`, which hides the local-image picker section.

### D5 — Drop the image-mode response branch in the frontend

`column_review/static/app.js` currently branches on
`data.tile_source_type === "image"` to mount OSD in image-mode.
After D1, the server never returns that key for the local-image
path. Removing the branch keeps the frontend tracking a single
"DZI tileSource" contract for both `/api/open` and
`/api/open-local-image`.

### D6 — Import shim, not subprocess

`scripts/ingest_drawings.py:158` exposes `ingest()` as a callable.
Routes already use
`from column_review.path_bootstrap import ensure_on_path` to import
sibling scripts in process. Spawning `python3 scripts/hitl.py ingest
...` as a subprocess would add ~1 s of import overhead and an
error-mapping hop for each click; the in-process call is cheaper
and lets us catch `FileNotFoundError`/`PermissionError`/`OSError`
and translate to typed HTTP errors directly.

## Risks / Trade-offs

- **First-click wall time** — A0 PNG ingest + DZI takes ~30–60 s on
  this hardware. → Frontend shows a spinner; banner if the call
  exceeds 90 s. Escalation path documented above (D2).
- **Partial-ingest crash leaves orphan files** — disk-full or PIL
  failure mid-DZI write could leave `<id>.png` without `<id>.dzi`. →
  Handler wraps the `ingest()` call and, on exception, deletes
  `<id>.{png,jpg,meta.json,dzi}` and `<id>_files/` under
  `data/raw/drawings/` before re-raising. Next click starts clean.
- **Drawing-id collision** — two source files with the same stem
  (e.g. `L3.png` and `L3.jpg`) would map to the same `drawing_id` and
  the second click could overwrite the first. → `ingest()` already
  calls `_delete_stale_siblings(drawing_id, keep=out_path)` which
  removes prior-suffix siblings during a rebuild; document this as
  expected behaviour and leave detection to the dataset hygiene step
  (out of scope).
- **`--images-dir` removal is BREAKING** — any user/script relying on
  the flag will fail at argparse. → Acceptable per user direction;
  call out in CLAUDE.md.

## Migration Plan

- Implementation steps live in `tasks.md`; deployment is a `git
  pull` + `pip install -e .` reinstall (the console-script entry
  point doesn't change). No DB migration. No model retrain. Rollback
  is `git revert` of this change's commit — no data is destroyed
  irreversibly by adopting the new flow.
