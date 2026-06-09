## 1. Backend: ingest-on-click endpoint

- [x] 1.1 In `column_review/routes/files.py`, add (at module top) `ensure_on_path("scripts")` plus `from ingest_drawings import ingest, resolve_drawing, INPUT_DPI` (re-use `column_review.path_bootstrap.ensure_on_path`).
- [x] 1.2 Refactor `post_open_local_image` so that after the existing path-safety + reviewer-id checks it derives `drawing_id = raster_path.stem` and calls `resolve_drawing(drawing_id)` to look for a pre-existing DZI.
- [x] 1.3 If `resolve_drawing` returns a meta with `dzi_path` set AND the file exists, skip the rebuild and fall through to the job + session bootstrap with that `drawing_id`.
- [x] 1.4 Otherwise wrap `ingest(raster_path, drawing_id, dpi=INPUT_DPI, build_tiles=True)` in `try / except OSError, PIL.UnidentifiedImageError, RuntimeError, SystemExit`. On exception, delete any partial `data/raw/drawings/<drawing_id>.{png,jpg,jpeg,tif,tiff,bmp,meta.json,dzi}` files and the `<drawing_id>_files/` directory, then `raise HTTPException(status_code=500, detail=f"ingest failed: {e}")`.
- [x] 1.5 Change the response shape: drop `tile_source_type: "image"`, set `tile_source = f"/tiles/{drawing_id}.dzi"`, and keep the other keys (`drawing_id`, `reviewer_id`, `job_id`, `session_id`, `detections_url`) unchanged.

## 2. Frontend: spinner + remove image-mode branch

- [x] 2.1 In `column_review/static/app.js` at the `/api/open-local-image` call site (around line 471), wrap the fetch in `withButtonSpinner(...)` against the picker open button so the user gets visible feedback during the 30–60 s ingest.
- [x] 2.2 Remove the response-side `if (data.tile_source_type === "image")` branch and the OSD `tileSources: { type: "image", url: ... }` configuration — the server now always returns a DZI `tile_source`, matching the `/api/open` path's response shape.
- [x] 2.3 If the fetch takes longer than 90 s, surface a one-line banner ("Ingest taking longer than expected, check the server log") via the existing `showFailBanner(...)` helper without aborting.

## 3. CLI: hard-wire watched folder, drop `--images-dir`

- [x] 3.1 In `column_review/cli.py`, remove the `--images-dir` argparse argument (around line 97-98).
- [x] 3.2 Replace the `images_dir` resolution block (around lines 130-137) with `images_dir = Path("~/Documents/retrain-dataset").expanduser()`, then guard with `if not images_dir.is_dir(): print("[warn] watched folder missing: ...", flush=True); images_dir = None`.
- [x] 3.3 Drop the prior `/home/jiezhi/Documents/PDF TGCH Floor Plan All` fallback (deleted by the rewrite in 3.2).

## 4. Docs

- [x] 4.1 Add a short subsection to `CLAUDE.md` under "Architecture" titled "Picker auto-ingest" noting (a) the click triggers a server-side `ingest_drawings.ingest(...)` call on first open, (b) the watched folder is hard-wired to `~/Documents/retrain-dataset/`, (c) the `--images-dir` flag is removed.
- [x] 4.2 If `README.md` mentions `--images-dir` or the `hitl.py ingest` step as a prerequisite to clicking, update it to point at the new flow.

## 5. Verification

- [ ] 5.1 Launch `column-review` (no flags) and confirm the picker drawer lists every PNG/JPG in `~/Documents/retrain-dataset/`. (Folder confirmed present with 71 files; needs interactive UI check.)
- [ ] 5.2 Click a previously-uningested file (e.g. an `_L4-` plan that isn't in `data/raw/drawings/` yet). Confirm the picker button shows a spinner and that `data/raw/drawings/<stem>.{png,meta.json,dzi}` + `<stem>_files/` appear during the call.
- [ ] 5.3 Confirm OSD opens with the new DZI tile source. Verify a `Run inference` click works on the new drawing end-to-end.
- [ ] 5.4 Click the same file again and confirm the response is sub-second (no rebuild).
- [ ] 5.5 Verify `/api/drawings` (DZI listing) now includes the new `drawing_id`.
- [x] 5.6 Pass `--images-dir /tmp` and confirm argparse rejects the unknown flag (sanity check that the flag is gone). (Verified — argparse exits with usage.)
- [ ] 5.7 Rename or temporarily move `~/Documents/retrain-dataset` and restart. Confirm the warning prints and the rest of the UI still loads.

## 6. Spec validation

- [x] 6.1 Run `openspec validate --change auto-ingest-folder-images` and confirm zero errors. (Validated — `Change 'auto-ingest-folder-images' is valid`.)
