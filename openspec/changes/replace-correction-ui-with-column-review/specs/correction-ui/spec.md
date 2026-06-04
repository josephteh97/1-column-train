## ADDED Requirements

### Requirement: Single-command launch

The system SHALL provide a single shell command `column-review` (registered as
a `[project.scripts]` console_script in `pyproject.toml`) that, after `pip
install -e .` or equivalent, MUST start a local FastAPI server and open the
user's default browser tab pointed at the UI. The command MUST work from any
current working directory.

If the configured default port (e.g., 8765) is already in use, the system
MUST automatically pick a free port and print the chosen URL to stdout. The
command MUST NOT fail because the default port is busy. There MUST NOT be a
separate frontend build step at run time.

#### Scenario: Launch from a non-project directory
- **WHEN** the user runs `column-review` from `/tmp` (a directory outside
  the project root)
- **THEN** the server starts, prints a line like `column-review listening
  on http://127.0.0.1:8765`, and opens that URL in the user's default
  browser

#### Scenario: Default port already in use
- **WHEN** the user runs `column-review` while another process is bound to
  the default port
- **THEN** the server picks the next free port, prints the chosen URL to
  stdout, and opens the browser to that URL (no crash, no manual port flag
  required)

### Requirement: File picker and open

The UI SHALL provide a file picker (or drawing-ID picker) that lets the
reviewer open any floor-plan image whose DZI tile pyramid has already been
built (under `data/raw/drawings/<id>.dzi` + `<id>_files/`). Supported source
raster formats are at minimum PNG, JPG, and TIFF. Selecting an entry MUST
load (or compute) detection boxes for that drawing and render the image
together with the detection overlay.

If the chosen drawing has no DZI pyramid on disk, the UI MUST surface a
typed error with the exact `python3 scripts/hitl.py ingest <path>`
invocation needed to build one, instead of stalling or rendering blank.

#### Scenario: Open a drawing that has a pre-built tile pyramid
- **WHEN** the reviewer selects a drawing entry whose
  `data/raw/drawings/<id>_files/` tree exists
- **THEN** the OpenSeadragon viewer mounts the tile pyramid, the detection
  overlay is populated from the model output, and both are visible within
  the 3-second budget

#### Scenario: Open a drawing whose tile pyramid is missing
- **WHEN** the reviewer selects a drawing whose `<id>_files/` directory
  does not exist
- **THEN** the UI shows a visible diagnostic banner naming the missing
  path and the exact `hitl.py ingest` command to fix it, without rendering
  a blank canvas

### Requirement: Image and detections MUST render together (R2 regression guard)

The system SHALL render BOTH the raster (via the DZI tile pyramid) AND all
model detection boxes on the page after a drawing is opened. A page state
where (a) the image renders but no detections appear, (b) detections render
but no image is loaded, or (c) the canvas is blank, MUST trigger a visible
fail banner with a diagnostic message identifying which side is missing —
NEVER a silent blank canvas.

Detection boxes MUST be anchored to the image's pixel coordinate system so
panning and zooming move image and boxes together with zero drift.

#### Scenario: Both layers render
- **WHEN** the reviewer opens a drawing
- **THEN** within 3 seconds the OpenSeadragon viewer has at least one
  tiled image source AND the detection overlay reports at least the same
  count of boxes as `data/jobs/<job_id>/px_detections.json` lists, AND no
  fail banner is shown

#### Scenario: Detections present, image missing
- **WHEN** the OpenSeadragon viewer has 0 tiled image sources at the
  point the canary fires (after `open` event + first detections fetch)
- **THEN** a visible fail banner is shown reading "Renderer state
  inconsistent: image missing" (or equivalent)

#### Scenario: Image present, detections missing
- **WHEN** the detections fetch returns an empty array and the renderer
  state canary fires
- **THEN** a visible fail banner is shown reading "Renderer state
  inconsistent: detections missing", NOT a silent empty overlay

#### Scenario: Pan/zoom drift
- **WHEN** the reviewer zooms in by 4× and pans to an arbitrary corner
- **THEN** the detection boxes remain centred on the same world-pixel
  coordinates as the underlying raster (no offset, no skew)

### Requirement: A0 tile-pyramid zoom and pan

The system SHALL render floor-plan images up to A0 at 300 DPI
(approximately 9933 × 14043 pixels, ~140 megapixels) using a pyramidal /
deep-zoom tile renderer (OpenSeadragon or equivalent). The full raster
MUST NOT be loaded as a single bitmap into memory or into the rendering
canvas. Open-to-first-render MUST be under 3 seconds on the deployment
hardware. Pan and zoom MUST target 60 fps at any zoom level.

The in-memory tile cache MUST be bounded with a configurable ceiling
(default 512 MB) enforced via LRU eviction. The ceiling MUST NOT be
exceeded by unbounded growth.

#### Scenario: A0 drawing opens within budget
- **WHEN** the reviewer opens a previously-ingested A0-at-300-DPI drawing
- **THEN** the first tile renders within 3 seconds of the open action

#### Scenario: Tile cache stays bounded
- **WHEN** the reviewer pans and zooms across a wide range of viewport
  positions until the LRU ceiling is reached
- **THEN** the tile cache memory stays at or below the configured
  ceiling, old tiles are evicted, and pan/zoom continues without OOM

### Requirement: Keyboard-first interaction

Every primary action SHALL have a single-key keyboard shortcut. The full
correction workflow (mark FP, draw FN, undo, redo, save-and-submit,
fit-to-window, 100% zoom, zoom-to-selection, next-detection,
previous-detection, jump-to-next-unreviewed) MUST be completable without
ever touching a menu or opening a modal dialog INSIDE the correction loop.
(Save & Submit may open a one-step confirmation modal because retraining
is outside the loop.)

#### Scenario: Complete a session with keyboard only
- **WHEN** the reviewer marks 100 detections, undoes 5, redoes 3, jumps to
  the next unreviewed, fits to window, zooms to 100%, and triggers Save &
  Submit — all via keyboard shortcuts only
- **THEN** every action completes and no menu or in-loop modal is required

### Requirement: Mouse interactions

The system SHALL support these mouse actions on the viewer canvas:
- Left-click on a detection box toggles its FP mark.
- Left-drag on empty space draws a new FN_ADDED bounding box; mouseup
  persists it as a single fluid action (no two-step "draw then confirm").
- Middle-drag OR Space+left-drag pans the viewport.
- Mouse-wheel zooms, centred on the cursor position (NOT on the viewport
  centre).

#### Scenario: Click toggles FP
- **WHEN** the reviewer left-clicks within the hit-test radius of an
  unreviewed detection
- **THEN** the detection's state flips to FP and the persisted row in
  `data/corrections.db` is written within 1 second

#### Scenario: Drag adds FN_ADDED
- **WHEN** the reviewer left-drags from `(x0, y0)` to `(x1, y1)` over
  empty (non-detection) canvas area and releases the mouse
- **THEN** a new detection with state FN_ADDED appears at that bounding
  box and the persisted correction row is written within 1 second

#### Scenario: Wheel zoom centred on cursor
- **WHEN** the reviewer points the cursor at a specific world-pixel
  position and scrolls the mouse wheel
- **THEN** the zoom is applied with the cursor position held fixed in
  screen-space (not the viewport centre)

### Requirement: Visual states distinct at all zoom levels

The system SHALL render three correction states with unambiguously
distinct visual treatments that remain distinguishable in the persistent
mini-map and at low zoom levels: `UNREVIEWED`, `MARKED_FP`, `FN_ADDED`.
An optional fourth state `MARKED_TP` MAY be implemented with a fourth
distinct treatment. Distinction MUST combine colour AND a minor border or
shape treatment (e.g., solid vs dashed vs dotted strokes) so the states
survive downsampling on the mini-map.

TP marking SHALL be optional — an unmarked detection is treated as
implicitly accepted (NOT as missing data). The standard workflow does
NOT require the reviewer to mark TPs.

#### Scenario: Three states at fit-to-window zoom
- **WHEN** the reviewer fits the drawing to the window and inspects 10
  detections each in UNREVIEWED, MARKED_FP, FN_ADDED states
- **THEN** each state is visually distinguishable from the other two
  without zooming in

### Requirement: Navigation aids

The system SHALL provide all of:
- A persistent mini-map showing the current viewport position relative
  to the full drawing AND the locations of unreviewed / FP / FN_ADDED
  clusters.
- A fit-to-window action (keyboard-bound).
- A 100% (1:1 pixel) zoom action (keyboard-bound).
- A zoom-to-selection action (keyboard-bound).
- A numeric zoom-level indicator that the reviewer can click to enter
  an exact percentage.
- A jump-to-next-unreviewed action that pans and zooms the viewport to
  the next detection still in UNREVIEWED state.

#### Scenario: Mini-map highlights remaining work
- **WHEN** the reviewer has marked half the detections in a drawing
- **THEN** the mini-map shows the unreviewed cluster locations
  distinctly from the marked-FP and FN_ADDED cluster locations

#### Scenario: Jump to next unreviewed
- **WHEN** the reviewer presses the jump-to-next-unreviewed shortcut
- **THEN** the viewport pans and zooms to centre on the next detection
  whose state is UNREVIEWED

### Requirement: Zoom-adaptive hit-testing

Clicking within a configurable pixel radius of a detection box SHALL
select it; pixel-perfect clicking MUST NOT be required. At low zoom
levels where boxes appear tiny on screen, the hit-test tolerance MUST
scale up automatically so the reviewer can still mark small boxes.

#### Scenario: Click near a tiny box at low zoom
- **WHEN** the reviewer is zoomed out such that a detection box appears
  3 px wide on screen and clicks within 6 px of the box centre
- **THEN** the box is selected (its FP mark toggles)

### Requirement: Undo and redo

The system SHALL support at least 100 levels of undo and redo per
session. Each undo or redo operation MUST be O(1). Z and Shift-Z (or
equivalent dedicated single-key shortcuts) MUST trigger them.

#### Scenario: Undo across 100 actions
- **WHEN** the reviewer makes 100 mark or draw actions and then presses
  Undo 100 times
- **THEN** every action is reversed in LIFO order and the session
  returns to its initial state

#### Scenario: Redo after undo
- **WHEN** the reviewer presses Undo 5 times and then Redo 3 times
- **THEN** the final 3 of the 5 undone actions are re-applied in order

### Requirement: Autosave on every action

Every correction action (mark FP, draw FN, undo, redo) SHALL be durable
to disk within 1 second of the action. A browser refresh or process
crash MUST NOT lose any persisted correction. The explicit Save & Submit
button is reserved for triggering retraining; saving is already done by
the time the reviewer reaches it.

#### Scenario: Browser refresh mid-session
- **WHEN** the reviewer marks 50 detections, then refreshes the browser
  tab before pressing Save & Submit
- **THEN** all 50 marks are present in the reopened session, loaded from
  `data/corrections.db`

### Requirement: Performance ceiling

The system MUST keep interaction lag (between a click or keypress and
the visible state-update on the canvas) under 50 ms on the deployment
hardware on A0-at-300-DPI drawings with up to 2000 detection boxes. If
the host machine cannot meet this budget, the system MUST report so
loudly at startup with a diagnostic identifying the bottleneck, rather
than degrade silently.

#### Scenario: Lag stays under 50ms with 2000 detections
- **WHEN** the reviewer opens an A0-at-300-DPI drawing with 2000
  detections and marks 100 FPs in rapid succession
- **THEN** the visible state-update for each click lands within 50 ms

#### Scenario: Hardware can't meet budget
- **WHEN** the system measures that a representative open-and-render
  cycle exceeds the budget at startup
- **THEN** a visible diagnostic banner reports the failure and names
  the bottleneck (e.g., "open-to-first-render took 4.2s; tile-pyramid
  build appears uncached"), instead of silently degrading

### Requirement: Save & Submit triggers retraining

The Save & Submit action SHALL persist the canonical corrections
snapshot (already auto-saved to `data/corrections.db`) and trigger the
retraining workflow with a structured contract: at minimum the
corrections-DB path, the drawing ID, and the reviewer ID. A
single-step confirmation dialog MUST be shown before the retraining
job is spawned, with the projected runtime and the correction counts
that will be folded in. Retraining MUST run as a `subprocess.Popen`
of `python3 scripts/retrain_yolo.py` on the local machine.

The retraining job MUST run in the background — the UI MUST NOT
synchronously block on it. The UI SHALL surface the job's lifecycle
state (`queued` / `running` / `completed` / `failed`) without blocking
further review work. A failed retraining MUST be visible to the
reviewer with the tail of stderr — silent failure is forbidden.

#### Scenario: Confirm dialog before retrain
- **WHEN** the reviewer presses Save & Submit with 27 FPs and 4 FN_ADDED
  marks in this session
- **THEN** a confirmation dialog appears showing the counts and the
  retrain command; the subprocess is only spawned if the reviewer
  confirms

#### Scenario: Status surfaced while retraining runs
- **WHEN** the retrain subprocess is mid-epoch
- **THEN** the UI shows a status pill reading "running" with the PID and
  the elapsed time, and review interactions remain responsive

#### Scenario: Retrain failure is visible
- **WHEN** the retrain subprocess exits with a non-zero return code
- **THEN** the UI status pill flips to "failed", the tail of stderr is
  shown in a banner the reviewer can dismiss, and the failure is
  recoverable by re-triggering Save & Submit

### Requirement: Forbidden patterns

The implementation MUST NOT exhibit any of the following:

- A page state where the image renders without detections, or
  detections render without the image, or the canvas is blank after a
  file is opened (this is the failure mode of the deleted page; the
  R2 regression guard MUST catch it).
- Loading the full A0 raster as a single bitmap into memory or canvas.
- A mouse-only workflow for any primary action — every primary action
  MUST have a single-key keyboard alternative.
- A modal dialog anywhere inside the correction loop. (The single
  confirm dialog for Save & Submit is outside the loop and permitted.)
- Loss of viewport zoom or pan state after any correction action.
- A "TP-required" workflow — TP marking MUST remain optional.
- Synchronous blocking of the UI while retraining runs.
- Silent failure of save or retrain — both MUST surface diagnostics.
- Pixel-perfect clicking required to mark a small detection.
- Any code path that re-exposes the deleted `scripts/correction_app/`
  UI behind a flag, environment variable, or config option.

#### Scenario: Zoom state preserved across a mark
- **WHEN** the reviewer is zoomed in at 220% on the bottom-right of the
  drawing and clicks to mark a detection FP
- **THEN** after the mark persists, the viewport remains at 220% zoom
  on the same bottom-right region (no reset to fit-to-window)

#### Scenario: Old page is not reachable
- **WHEN** the reviewer (or a script) attempts `python3 scripts/hitl.py
  review <drawing-id>`
- **THEN** argparse rejects the subcommand (`invalid choice: 'review'`)
  and no fallback path serves the old UI

### Requirement: Corrections output contract

The system SHALL persist corrections to `data/corrections.db` (SQLite)
using the existing schema owned by `scripts/corrections_logger.py`:
the `corrections` table (`id`, `job_id`, `element_type`,
`element_index`, `original_element`, `changes`, `is_delete`,
`timestamp`) with `UNIQUE INDEX (job_id, element_index, is_delete)`,
plus sidecar tables `tp_confirmations (session_id, job_id,
element_index, ts)` and `reviewer_sessions (session_id, reviewer_id,
started_ts)`.

The encoding MUST be:
- `MARKED_FP` → a row with `is_delete=1` against the source
  `element_index` from `px_detections.json`. The rescind invariant
  (writing an `is_delete=0` row for the same `(job_id,
  element_index)` cancels the FP) MUST be respected by undo.
- `FN_ADDED` → the drawn bbox is **appended to
  `data/jobs/<job_id>/px_detections.json["columns"]`** as a new
  entry tagged `"source": "human_added"`. A corresponding
  `corrections` row with `is_delete=0` is written using the
  newly-assigned positive `element_index`. This mirrors the
  on-disk shape that `scripts/retrain_yolo.py` already consumes:
  retrain reads bboxes solely from `px_detections.json` and uses
  the corrections rows as a delete-set keyed by `(element_type,
  element_index)`. Any FN_ADDED encoding that does NOT append to
  the JSON would be silently dropped by retrain.
- `MARKED_TP` (optional) → a row in `tp_confirmations` keyed by
  `(job_id, element_index)`. Absence of a row means
  implicitly-accepted, not "no data".

The new package MUST take ownership of the `CREATE TABLE IF NOT EXISTS`
statements for `tp_confirmations` and `reviewer_sessions` (formerly
issued by the deleted `correction_app/app.py`). The
`scripts/retrain_yolo.py` consumer of `data/corrections.db` MUST NOT
require any change — it reads the same tables as today.

#### Scenario: FP persists with the right encoding
- **WHEN** the reviewer marks detection index `42` as FP for `job_id =
  J`
- **THEN** a row `(job_id=J, element_index=42, is_delete=1)` is
  written to `corrections` within 1 second

#### Scenario: FN_ADDED appends to the JSON and writes a positive-index row
- **WHEN** the reviewer drags a new bbox during the same session
- **THEN** a new entry `{"bbox": [...], "source": "human_added"}`
  is appended to `data/jobs/<job_id>/px_detections.json["columns"]`,
  AND a `corrections` row with `is_delete=0` is written using the
  newly-assigned positive `element_index` (the index of the new
  entry in the JSON columns list)

#### Scenario: Undo of FP rescinds via is_delete=0
- **WHEN** the reviewer marks index `7` as FP then presses Undo
- **THEN** a follow-up row `(job_id=J, element_index=7, is_delete=0)`
  is written and `iter_effective_corrections` no longer reports
  index 7 as deleted

### Requirement: Old correction surface deleted in this change

The `scripts/correction_app/` package (8 files, ~2,440 LOC) MUST be
removed from the codebase in this change. The `review` subcommand of
`scripts/hitl.py` MUST be removed in this same change. There MUST NOT
be a parallel "legacy mode" or "classic mode" toggle, a feature flag,
an environment variable, or a config option that re-exposes the old
UI. There MUST NOT be a transition period during which both UIs
coexist.

All user-facing documentation in this repository
(`README.md`, `CLAUDE.md`) that referenced the old surface MUST be
updated in this same change to invoke `column-review` instead.

#### Scenario: Old package is gone
- **WHEN** the change is applied
- **THEN** `scripts/correction_app/` does not exist on disk, and no
  module elsewhere in the repo imports `scripts.correction_app`

#### Scenario: Old CLI subcommand is gone
- **WHEN** the change is applied and a user runs `python3
  scripts/hitl.py --help`
- **THEN** `review` is not listed among the available subcommands,
  while `ingest`, `build-tiles`, `retrain`, and `status` still are
