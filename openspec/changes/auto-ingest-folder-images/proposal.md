## Why

Opening a new drawing in `column-review` currently requires the user
to drop the file in the right place, then run
`python3 scripts/hitl.py ingest <plan> --drawing-id <id>` on the CLI,
and only then click the drawing in the web picker. The CLI hop is
friction for the intended workflow ("drop image → start reviewing")
and is the only manual step left between dropping a file in the
watched folder and producing corrections. Every retrain image now
lives in `~/Documents/retrain-dataset/` (71 files today), and the
user wants the picker to serve only that folder and auto-ingest on
the first click.

## What Changes

- **MODIFY** `POST /api/open-local-image` in
  `column_review/routes/files.py` so that clicking a file in the
  picker calls `scripts.ingest_drawings.ingest(...)`
  (with `build_tiles=True`) on the first open and returns a DZI
  `tile_source` (`/tiles/<drawing_id>.dzi`) — not the raw-image
  tileSource it returns today. Re-clicks for the same `drawing_id`
  short-circuit via `resolve_drawing(...)` and skip the rebuild.
- **MODIFY** the picker click handler in `column_review/static/app.js`
  to (a) show a spinner during the synchronous ingest round-trip and
  (b) drop the `tile_source_type: "image"` branch since the server
  now always returns a DZI source.
- **MODIFY** `column_review/cli.py` to hard-wire `images_dir` to
  `~/Documents/retrain-dataset` (expanded via `Path.expanduser()`)
  and **remove the `--images-dir` argparse flag** plus its prior
  `/home/jiezhi/Documents/PDF TGCH Floor Plan All` fallback. The
  launch command stays exactly `column-review` — no new flags.
- **BREAKING**: `--images-dir` is removed. Anyone relying on it
  must drop files into `~/Documents/retrain-dataset/` instead.

## Capabilities

### New Capabilities
<!-- none — this is a behavioral change to existing capabilities -->

### Modified Capabilities
- `correction-ui`: the local-image picker now triggers a server-side
  ingest on the first click instead of opening in raw-image mode;
  the watched folder is hard-wired to `~/Documents/retrain-dataset/`.
- `drawing-tile-pyramid`: ingestion can now be triggered by a web
  click on a file under `images_dir`, in addition to the existing
  `hitl.py ingest` CLI path. The DZI write contract itself is
  unchanged.

## Impact

- **Code**: `column_review/routes/files.py` (endpoint behavior),
  `column_review/cli.py` (flag removal + hard-wired folder),
  `column_review/static/app.js` (picker click handler), `CLAUDE.md`
  (one-line architecture note).
- **APIs**: `POST /api/open-local-image` response shape narrows
  (drops `tile_source_type`); request shape unchanged.
- **Dependencies**: none added. Already-imported `Pillow` +
  `scripts.ingest_drawings` cover the new path.
- **Data on disk**: each first-click writes
  `data/raw/drawings/<stem>.{png|jpg|...}`, `<stem>.meta.json`,
  `<stem>.dzi`, and `<stem>_files/` — same artefacts the CLI ingest
  already produces. No DB schema change.
- **Out of scope**: background-job pattern for ingest, inotify-style
  folder watching, configurable DPI, multi-page picker.
