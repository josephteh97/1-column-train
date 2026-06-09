## ADDED Requirements

### Requirement: Web-triggered ingestion via picker click

The system SHALL allow a `POST /api/open-local-image` call to trigger the same `ingest_drawings.ingest(source, drawing_id, dpi, build_tiles=True)` pipeline that the `python3 scripts/hitl.py ingest` CLI command invokes today, when the file the user clicked has no pre-existing DZI on disk for its derived `drawing_id`. The web-triggered path SHALL produce the SAME on-disk artefact set (`data/raw/drawings/<id>.<ext>`, `<id>.meta.json`, `<id>.dzi`, `<id>_files/`) and SHALL use the SAME DZI tile layout, JPEG quality, and level count as the CLI-triggered path. The two ingest entry points SHALL be byte-equivalent for the same source file.

#### Scenario: Web-click ingest produces the same DZI as CLI ingest

- **WHEN** a file `L4.png` is first opened via the picker
- **AND** `data/raw/drawings/L4.dzi` did not previously exist
- **THEN** `data/raw/drawings/L4.{png,meta.json,dzi}` and `data/raw/drawings/L4_files/` SHALL be created
- **AND** the DZI XML manifest's `<Size>` SHALL match the canonical raster size
- **AND** the tiles SHALL be JPEG quality 80 at 256-px edges with 1-px overlap, matching `### Requirement: Tile layout and JPEG format`
- **AND** the level count SHALL be `floor(log2(max(W, H)))` matching the CLI ingest path

#### Scenario: Web-click ingest is idempotent

- **WHEN** the user clicks an already-ingested file
- **THEN** the server SHALL NOT re-run `ingest_drawings.ingest(...)`
- **AND** SHALL NOT rewrite any file under `data/raw/drawings/`

#### Scenario: Web-click ingest uses the package default DPI

- **WHEN** a web-triggered ingest runs against an image without embedded DPI metadata
- **THEN** the resulting `<id>.meta.json` SHALL record the same `dpi` value the CLI path's `INPUT_DPI` default produces
- **AND** the DZI pixel content SHALL match what the CLI path would have produced for the same source

## MODIFIED Requirements

### Requirement: DZI tile pyramid generation at ingestion time

The system SHALL extend `scripts/ingest_drawings.py` so that every successfully ingested drawing produces a Microsoft Deep Zoom Image (DZI) tile pyramid alongside the existing canonical raster and metadata. The DZI manifest MUST be written to `data/raw/drawings/<drawing-id>.dzi` and the tile JPEGs MUST be written under `data/raw/drawings/<drawing-id>_files/<level>/<col>_<row>.jpg`. Generation MUST execute inline as part of `ingest_drawings.py` and MUST run after the raster is finalised so the DZI matches the canonical raster's pixel content exactly. Ingestion MAY be triggered either by the CLI command `python3 scripts/hitl.py ingest <plan> --drawing-id <id>` or by a web click on a file under the column-review picker's hard-wired watched folder (`~/Documents/retrain-dataset/`).

#### Scenario: A fresh CLI ingest produces the DZI

- **WHEN** the reviewer runs `python3 scripts/hitl.py ingest <plan-path> --drawing-id <id>`
- **AND** the underlying call to `scripts/ingest_drawings.py` succeeds
- **THEN** `data/raw/drawings/<id>.dzi` exists with a Microsoft DZI XML manifest
- **AND** `data/raw/drawings/<id>_files/` exists with one subdirectory per pyramid level
- **AND** the highest-level (largest) tiles cover the full pixel extent of the canonical raster

#### Scenario: A fresh web-click ingest produces the DZI

- **WHEN** the reviewer clicks a previously-uningested file in the column-review picker
- **AND** the server-side ingest call succeeds
- **THEN** `data/raw/drawings/<drawing-id>.dzi` exists with a Microsoft DZI XML manifest
- **AND** `data/raw/drawings/<drawing-id>_files/` exists with one subdirectory per pyramid level
- **AND** the artefacts are byte-equivalent to what a CLI ingest of the same source would have produced

#### Scenario: DZI matches the canonical raster pixel-for-pixel at level max

- **WHEN** the highest-resolution level of the DZI is reassembled from its tiles
- **THEN** the reassembled image matches `data/raw/drawings/<id>.<ext>` pixel-for-pixel
- **AND** the DZI manifest's `<Size Width="..." Height="..."/>` matches the canonical raster size
