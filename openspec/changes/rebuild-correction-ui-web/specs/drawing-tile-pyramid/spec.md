## ADDED Requirements

### Requirement: DZI tile pyramid generation at ingestion time

The system SHALL extend `scripts/ingest_drawings.py` so that every successfully ingested drawing produces a Microsoft Deep Zoom Image (DZI) tile pyramid alongside the existing canonical raster and metadata. The DZI manifest MUST be written to `data/raw/drawings/<drawing-id>.dzi` and the tile JPEGs MUST be written under `data/raw/drawings/<drawing-id>_files/<level>/<col>_<row>.jpg`. Generation MUST execute inline as part of `ingest_drawings.py` and MUST run after the raster is finalised so the DZI matches the canonical raster's pixel content exactly.

#### Scenario: A fresh ingest produces the DZI

- **WHEN** the reviewer runs `python3 scripts/hitl.py ingest <plan-path> --drawing-id <id>`
- **AND** the underlying call to `scripts/ingest_drawings.py` succeeds
- **THEN** `data/raw/drawings/<id>.dzi` exists with a Microsoft DZI XML manifest
- **AND** `data/raw/drawings/<id>_files/` exists with one subdirectory per pyramid level
- **AND** the highest-level (largest) tiles cover the full pixel extent of the canonical raster

#### Scenario: DZI matches the canonical raster pixel-for-pixel at level max

- **WHEN** the highest-resolution level of the DZI is reassembled from its tiles
- **THEN** the reassembled image matches `data/raw/drawings/<id>.<ext>` pixel-for-pixel
- **AND** the DZI manifest's `<Size Width="..." Height="..."/>` matches the canonical raster size

### Requirement: Tile layout and JPEG format

The system SHALL emit tiles of edge size 256 pixels (with a 1-pixel overlap as per the DZI specification) at JPEG quality 80 in RGB. The pyramid SHALL contain levels from `0` (a 1x1 tile representing the entire drawing) up through `floor(log2(max(W, H)))` where W and H are the canonical raster width and height. No tile JPEG file SHALL exceed 200 kilobytes in expected size for a typical floor-plan tile.

#### Scenario: Tile dimensions and overlap

- **WHEN** any tile in the pyramid is inspected
- **THEN** its decoded image is 256 pixels per side (or less at the bottom-right edges) with the standard DZI 1-pixel overlap
- **AND** its format is JPEG at quality 80

#### Scenario: Level count derived from raster size

- **WHEN** the canonical raster is W=9933, H=14043 (A0 at 300 DPI)
- **THEN** the pyramid contains levels 0 through 14 inclusive
- **AND** level 0 contains a single 1x1 tile
- **AND** level 14 contains tiles covering the full 9933x14043 extent

### Requirement: Backfill subcommand for previously-ingested drawings

The system SHALL provide a `python3 scripts/hitl.py build-tiles <drawing-id>` subcommand that generates the DZI tile pyramid for an already-ingested drawing whose `data/raw/drawings/<id>.<ext>` and `data/raw/drawings/<id>.meta.json` exist but whose DZI does not. The subcommand MUST be idempotent: re-running on a drawing whose DZI already exists MUST overwrite the pyramid without prompting and without partial state.

#### Scenario: Backfill creates the DZI for an existing drawing

- **WHEN** the canonical raster `data/raw/drawings/<id>.png` exists but `data/raw/drawings/<id>.dzi` does not
- **AND** the reviewer runs `python3 scripts/hitl.py build-tiles <id>`
- **THEN** the DZI manifest and tile tree are created as specified above

#### Scenario: Backfill is idempotent

- **WHEN** the DZI for `<id>` already exists
- **AND** the reviewer runs `python3 scripts/hitl.py build-tiles <id>` again
- **THEN** the previous DZI tree is fully replaced with a freshly generated one
- **AND** no intermediate or partial pyramid state remains on disk if the command succeeds

### Requirement: Opt-out flag with downstream contract

The system SHALL accept an optional `--no-tiles` flag on `scripts/ingest_drawings.py` (and therefore on `python3 scripts/hitl.py ingest`) that skips DZI generation for fast non-review use cases. When the flag is used, the canonical raster and metadata MUST still be written exactly as before. The correction web reviewer MUST refuse to open such a drawing and MUST display a single diagnostic message directing the reviewer at the backfill subcommand. No silent fallback to single-bitmap rendering SHALL occur.

#### Scenario: --no-tiles skips DZI but keeps raster

- **WHEN** the reviewer runs `python3 scripts/hitl.py ingest <plan> --drawing-id <id> --no-tiles`
- **THEN** `data/raw/drawings/<id>.<ext>` and `data/raw/drawings/<id>.meta.json` exist
- **AND** `data/raw/drawings/<id>.dzi` does not exist

#### Scenario: Reviewer refuses to open a no-DZI drawing

- **WHEN** the reviewer runs `python3 scripts/hitl.py review <id>` for a drawing whose DZI is absent
- **THEN** the web UI displays a single diagnostic message naming the missing DZI and instructing the reviewer to run `python3 scripts/hitl.py build-tiles <id>`
- **AND** the UI does not attempt to load the canonical raster as a single bitmap fallback

### Requirement: Disk-cost transparency

The system SHALL document, in `READMD.md` and in the `--help` output of `scripts/hitl.py ingest` and `scripts/hitl.py build-tiles`, that DZI tile generation costs approximately 25 to 35 percent of additional disk per drawing on top of the canonical raster.

#### Scenario: Documented in --help

- **WHEN** the reviewer runs `python3 scripts/hitl.py ingest --help` or `python3 scripts/hitl.py build-tiles --help`
- **THEN** the help text contains the disk-cost note and points at `READMD.md` for details

#### Scenario: Documented in READMD.md

- **WHEN** the reviewer reads `READMD.md`
- **THEN** the disk-cost note is present near the ingest/review workflow documentation
