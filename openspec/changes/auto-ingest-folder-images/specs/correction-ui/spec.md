## ADDED Requirements

### Requirement: Picker auto-ingests on first click and serves a single hard-wired folder

The `column-review` UI SHALL display a "Local images" section in the picker drawer that lists all PNG/JPG/JPEG/TIFF/BMP files in `~/Documents/retrain-dataset/` (expanded via `Path.expanduser()`). This folder SHALL be the SOLE source for that picker section: no command-line flag SHALL override it, and no other folder SHALL ever populate that list. Selecting a file from this section SHALL cause the server to invoke the drawing-ingest pipeline (rasterise if needed + DZI tile pyramid build) on the first open and SHALL return a DZI `tile_source` (`/tiles/<drawing_id>.dzi`) for OpenSeadragon to mount. A subsequent click on the same file SHALL skip the rebuild via the existing on-disk DZI and open near-instantly. The launch command SHALL remain exactly `column-review` with no new arguments.

#### Scenario: First click on a fresh file triggers ingest

- **WHEN** the user drops `L4-new.png` into `~/Documents/retrain-dataset/`
- **AND** clicks `L4-new.png` in the picker
- **THEN** the server invokes `ingest_drawings.ingest(..., build_tiles=True)` with `drawing_id = "L4-new"`
- **AND** writes `data/raw/drawings/L4-new.{png,meta.json,dzi}` and `data/raw/drawings/L4-new_files/`
- **AND** the response carries `tile_source = "/tiles/L4-new.dzi"`
- **AND** the OSD viewer mounts the DZI tile source and opens the drawing

#### Scenario: Re-click on an already-ingested file is idempotent

- **WHEN** the user clicks a file whose DZI already exists on disk
- **THEN** the server SHALL NOT rebuild the DZI
- **AND** SHALL return the same `tile_source = "/tiles/<drawing_id>.dzi"` shape as a first-click response
- **AND** the response SHALL arrive within the same wall-time budget as a `POST /api/open` call on a DZI-ingested drawing

#### Scenario: `--images-dir` flag is removed

- **WHEN** the user runs `column-review --images-dir /some/other/path`
- **THEN** argparse SHALL reject the unknown argument
- **AND** the picker SHALL continue to list ONLY files from `~/Documents/retrain-dataset/`

#### Scenario: Watched folder missing

- **WHEN** `~/Documents/retrain-dataset/` does not exist at server start
- **THEN** the server SHALL log a one-line warning
- **AND** the "Local images" picker section SHALL be hidden or empty
- **AND** the rest of the UI SHALL load and function (DZI-ingested drawings still openable via `POST /api/open`)

#### Scenario: Mid-ingest failure does not leave orphan artefacts

- **WHEN** an ingest call fails partway through (e.g. disk full while writing DZI tiles)
- **THEN** the server SHALL delete any partial `data/raw/drawings/<drawing_id>.{png,jpg,meta.json,dzi}` files and the `<drawing_id>_files/` directory before responding
- **AND** SHALL return a structured 500 with `detail = "ingest failed: <reason>"`
- **AND** a subsequent click on the same file SHALL start cleanly without manual cleanup

## MODIFIED Requirements

### Requirement: Single-drawing scope per session

The system SHALL open exactly one drawing per running process, identified by the file the reviewer clicks in the picker (`POST /api/open` for DZI-ingested drawings, `POST /api/open-local-image` for files under `~/Documents/retrain-dataset/`). The UI MUST NOT include a drawing switcher or in-session drawing-change action. Switching drawings requires closing the picker, choosing a different file, and bootstrapping a new session.

#### Scenario: One drawing per opened session

- **WHEN** the reviewer opens `TGCH-TD-S-200-L3-00` via the picker
- **THEN** the served UI displays only that drawing
- **AND** no in-UI control allows switching to a different drawing without returning to the picker

#### Scenario: Switching drawings goes back through the picker

- **WHEN** the reviewer wishes to review a second drawing
- **THEN** they reopen the picker drawer and click a different file
- **AND** the server bootstraps a fresh session bound to the new `drawing_id`
