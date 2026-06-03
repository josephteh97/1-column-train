## ADDED Requirements

### Requirement: Wholesale replacement of the Jupyter correction UI

The system SHALL delete `correct_detections.ipynb` from the repository in this change and MUST NOT keep, gate, or expose any code path that re-enables the notebook. No environment variable, config key, feature flag, command-line argument, or runtime branch SHALL revive the notebook flow. The new web reviewer is the sole correction-marking surface from this change forward.

#### Scenario: Notebook is absent from the repository

- **WHEN** any reviewer or contributor inspects the repository root after this change is applied
- **THEN** `correct_detections.ipynb` does not exist at the repository root
- **AND** no other file in the repository imports, references, or links to `correct_detections.ipynb` as a working UI

#### Scenario: No legacy-mode toggle exists

- **WHEN** a reviewer searches the codebase for references to "classic", "legacy", "old-ui", or "notebook-ui" config keys or environment variables
- **THEN** no such code path exists that launches the deleted notebook
- **AND** `python3 scripts/hitl.py review <drawing-id>` is the only invocation that opens a correction-marking interface

### Requirement: Keyboard-first marking workflow

The system SHALL bind every primary marking action to a single-key shortcut and MUST allow a reviewer to complete the entire marking workflow on one drawing without touching a menu, modal dialog, or right-click context action. Primary actions covered: mark TP, mark FP, delete, add bounding box for missed column, undo, redo, next unreviewed, previous unreviewed, zoom in, zoom out, pan, fit-to-window, 100% zoom, save.

#### Scenario: Single-key shortcut per primary action

- **WHEN** the reviewer presses the bound key for any primary action (e.g. `T` for mark TP, `F` for mark FP, `D` for delete, `A` for begin-add-FN, `U` for undo, `Y` for redo, `N` for next unreviewed, `P` for previous unreviewed, `+`/`-` for zoom, Space-hold for pan, `F` for fit, `0` for 100%)
- **THEN** the action executes within one frame
- **AND** no modal dialog, popup, or confirmation prompt appears at any point in the action

#### Scenario: Marking a full drawing without leaving the keyboard

- **WHEN** a reviewer opens a drawing with N detections and marks every detection as TP or FP using only keyboard input
- **THEN** the session completes with zero mouse clicks on menus, toolbars, or modal buttons
- **AND** all N detections have a persisted mark in `data/corrections.db` or the sidecar `tp_confirmations` table

### Requirement: Pan and zoom do not fight marking actions

The system SHALL treat a left-click on a detection bounding box as a marking action (select / focus that detection) and MUST route panning to a separate input: middle-mouse-drag or a held modifier (Space for pan-grab). Mouse-wheel zoom SHALL center on the cursor position. Keyboard `+`/`-` zoom SHALL center on the current selection if any is active, otherwise on the cursor position.

#### Scenario: Left-click selects, does not pan

- **WHEN** the reviewer left-clicks on a detection box
- **THEN** the box becomes the active selection (highlighted) and is eligible for keyboard marking
- **AND** the viewport does not pan

#### Scenario: Wheel zoom centers on cursor

- **WHEN** the reviewer scrolls the mouse wheel over a point P in the viewport
- **THEN** the viewport zooms in or out
- **AND** the world point under cursor P before and after the zoom is the same world point (zoom anchored on cursor)

#### Scenario: Space-drag pans without marking

- **WHEN** the reviewer holds Space and drags the mouse
- **THEN** the viewport pans by the drag delta
- **AND** no detection is selected, marked, or deleted by the drag

### Requirement: Tile-based rendering of large drawings

The system SHALL render the underlying drawing as a pyramidal/multi-resolution tile pyramid using the deep-zoom (DZI) pattern. The full-resolution raster MUST NOT be loaded into memory or onto a canvas as a single bitmap at any time. On open, only the tiles for the current viewport at the appropriate level of detail SHALL be fetched.

#### Scenario: A0/300DPI drawing opens in under 3 seconds

- **WHEN** the reviewer runs `python3 scripts/hitl.py review <drawing-id>` for a drawing whose canonical raster is approximately 9933 x 14043 pixels (A0 at 300 DPI)
- **THEN** the viewer is responsive (the lowest-resolution tiles are rendered and pan/zoom is functional) within 3 seconds of the browser navigating to the served page
- **AND** at no point is the full raster decoded into a single in-memory bitmap

#### Scenario: Pan and zoom remain smooth across zoom levels

- **WHEN** the reviewer pans or zooms continuously over a 5-second interval on an A0/300DPI drawing with 1500 overlaid detection boxes
- **THEN** the rendering loop maintains at least 55 fps averaged across the interval at every visited zoom level
- **AND** no frame exceeds 50 ms of work on the main thread

### Requirement: Configurable tile cache memory ceiling with LRU eviction

The system SHALL expose a configurable memory ceiling for the rendered tile cache, defaulting to 512 megabytes. When the cache exceeds the ceiling, the system MUST evict least-recently-used tiles down to the ceiling. The cache MUST NOT grow unbounded under any access pattern.

#### Scenario: Cache stays at or below the configured ceiling

- **WHEN** the reviewer pans across the full extent of an A0/300DPI drawing such that the working tile set exceeds the configured ceiling (e.g. 512 MB)
- **THEN** the in-memory tile cache occupies no more than 512 megabytes at any sampled moment
- **AND** previously-loaded tiles outside the recent working set have been evicted in least-recently-used order

#### Scenario: Ceiling is overridable

- **WHEN** the reviewer launches `python3 scripts/hitl.py review <drawing-id> --tile-cache-mb 256`
- **THEN** the tile cache enforces a 256 megabyte ceiling for the session

### Requirement: Navigation aids for large drawings

The system SHALL display a persistent mini-map showing the current viewport position relative to the full drawing extent, and MUST provide single-key shortcuts for fit-to-window, 100% (1:1 pixel) zoom, zoom-to-selection, and reset-view. A numeric zoom-level indicator MUST be visible and clickable so the reviewer can type an exact percentage. The mini-map MUST highlight the locations of clusters of unreviewed detections so the reviewer can navigate visually to remaining work.

#### Scenario: Mini-map reflects viewport and unreviewed clusters

- **WHEN** the reviewer pans or zooms the main viewport
- **THEN** the mini-map's viewport-rectangle overlay updates in real time to reflect the new position and size relative to the full drawing
- **AND** the mini-map shows colour-coded dots or marks at the world positions of unreviewed detections so that clusters are visible at the mini-map's downsampled scale

#### Scenario: Numeric zoom indicator accepts exact input

- **WHEN** the reviewer clicks the zoom-level indicator and types "75" followed by Enter
- **THEN** the viewport zooms to exactly 75% of the 1:1 pixel scale
- **AND** the indicator reads "75%"

#### Scenario: Fit, 100%, zoom-to-selection, reset-view shortcuts

- **WHEN** the reviewer presses the bound key for fit-to-window, 100%, zoom-to-selection, or reset-view
- **THEN** the viewport transitions to the corresponding state within one frame

### Requirement: Four unambiguous visual states

The system SHALL render each detection in exactly one of four visual states — unreviewed, marked-TP, marked-FP, marked-FN-added — and MUST distinguish them with four distinct colours that have sufficient contrast on both light and dark drawing backgrounds. The four states MUST ALSO carry a minor shape or border treatment (e.g. solid vs dashed border, filled corner badge, stroke thickness) so that they remain distinguishable on the mini-map after downsampling and for colour-blind reviewers.

#### Scenario: Four states are visually distinct in the main viewport

- **WHEN** the viewport contains at least one detection in each of the four states
- **THEN** a reviewer with normal vision identifies each detection's state at a glance without hovering or selecting it
- **AND** the four colours pass a WCAG AA contrast ratio against both white and black backgrounds

#### Scenario: States survive downsampling on the mini-map

- **WHEN** the mini-map displays detections that are smaller than 3 px on screen
- **THEN** the four states remain visually distinguishable through their shape or border treatment, not colour alone

### Requirement: Undo and redo with at least 100 levels and O(1) per action

The system SHALL maintain an undo stack of at least 100 marking actions and MUST make every push, pop, and re-apply operation O(1) in time and memory per action. Undo and redo MUST cover: mark TP, mark FP, delete, add FN, batch operations, and movement of an added FN box before commit.

#### Scenario: 100 actions are reversible

- **WHEN** the reviewer performs 100 marking actions and then presses U (undo) 100 times
- **THEN** all 100 actions are reverted and the drawing's correction state returns to its pre-session state

#### Scenario: Undo is O(1) per action

- **WHEN** the reviewer presses U (undo) on a stack of any depth between 1 and 100
- **THEN** the action completes in under 5 milliseconds of main-thread time

#### Scenario: Redo recovers undone actions

- **WHEN** the reviewer undoes 5 actions and then presses Y (redo) 5 times
- **THEN** the 5 actions are re-applied in original order and the correction state matches the pre-undo state

### Requirement: Zoom-adaptive hit-test tolerance for small boxes

The system SHALL accept a left-click within a tolerance radius of a detection bounding box as a selection of that box. The tolerance MUST be configurable in CSS pixels and MUST automatically scale up at low zoom levels so the reviewer can interact with detections that appear smaller than 8 CSS pixels on screen. Pixel-perfect clicking MUST NOT be required at any zoom level.

#### Scenario: Small box on screen still hits

- **WHEN** the viewport is zoomed out such that a detection box subtends 4 CSS pixels of screen area
- **AND** the reviewer left-clicks 6 CSS pixels outside the box's nearest edge
- **THEN** the box is selected

#### Scenario: Tolerance is configurable

- **WHEN** the reviewer launches with `--hit-tolerance-px 16`
- **THEN** the base hit-test tolerance is 16 CSS pixels at 100% zoom and scales up at lower zoom levels

### Requirement: Single-drag FN addition

The system SHALL allow the reviewer to add a missed-column bounding box (FN_ADDED) by holding the add-FN modifier or pressing the add-FN key and performing one continuous mouse drag. On mouse-up the box is committed and durably written within 1 second. There MUST NOT be a separate confirm-or-cancel step or a modal preview before commit.

#### Scenario: Drag-to-add commits on mouse-up

- **WHEN** the reviewer presses A (begin add) and drags from world point (x1,y1) to (x2,y2) then releases the mouse button
- **THEN** a new FN_ADDED detection with bbox [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)] is recorded
- **AND** the box is durably persisted to `data/jobs/{job_id}/px_detections.json` within 1 second of the mouse-up event
- **AND** no confirmation dialog, modal, or two-step wizard appears between mouse-up and persistence

#### Scenario: Optional snap-to-grid

- **WHEN** snap-to-grid is enabled and the reviewer drag-adds an FN box
- **THEN** the box corners snap to the configured grid spacing on commit

### Requirement: Batch operations on selections

The system SHALL provide a rubber-band selection gesture (e.g. Shift+drag) that selects every detection whose bounding box intersects the selection rectangle, and MUST provide batch actions to mark all selected detections as FP and to delete all selected detections.

#### Scenario: Rubber-band select intersecting boxes

- **WHEN** the reviewer holds Shift and drags a rectangle that intersects 25 detection bounding boxes
- **THEN** all 25 detections become selected
- **AND** the selection count is visible in the progress UI

#### Scenario: Mark-all-in-selection as FP

- **WHEN** 25 detections are selected and the reviewer presses the batch-mark-FP key
- **THEN** all 25 detections are recorded as FP corrections
- **AND** all 25 writes are durable within 1 second

#### Scenario: Delete-all-in-selection

- **WHEN** 25 detections are selected and the reviewer presses the batch-delete key
- **THEN** all 25 detections are removed from the active set with corresponding rescind/delete records

### Requirement: Persistent progress UI and jump-to-next-unreviewed

The system SHALL display live counts of detections in each of the four states (unreviewed, TP, FP, FN_ADDED). The system MUST provide a jump-to-next-unreviewed action bound to N (and a previous-unreviewed action bound to P) that pans and zooms the viewport to the next unreviewed detection. The system MUST provide a filter mode that shows only detections in one selected state at a time.

#### Scenario: Counts update on every action

- **WHEN** the reviewer marks a previously-unreviewed detection as TP
- **THEN** the unreviewed count decreases by 1 and the TP count increases by 1, both reflected in the progress UI within one frame

#### Scenario: Jump-to-next-unreviewed pans and zooms

- **WHEN** the reviewer presses N (next unreviewed)
- **THEN** the viewport pans and zooms to bring the next unreviewed detection (in a deterministic order, e.g. row-major) to viewport centre at a zoom level where the detection subtends at least 80 CSS pixels

#### Scenario: Filter-by-state shows one category

- **WHEN** the reviewer activates the filter-by-state mode and selects "FP"
- **THEN** only FP detections are rendered in the overlay; the other three states are hidden

### Requirement: Autosave per action with at-most-1-second durability

The system SHALL persist every marking action to disk within 1 second of its commit. There MUST NOT be a separate "save" key, button, menu item, or close-time prompt. On unexpected close or process crash, no committed action is lost.

#### Scenario: Action is durable within 1 second

- **WHEN** the reviewer marks a detection as FP at time T
- **THEN** by time T + 1 second the corresponding row exists in `data/corrections.db` with `is_delete=1`

#### Scenario: No save UI exists

- **WHEN** the reviewer inspects the UI for a save button, save menu, or save shortcut
- **THEN** none exists; the act of marking is itself the act of saving

#### Scenario: Crash does not lose committed actions

- **WHEN** the reviewer marks 50 detections and the process is killed (SIGKILL) at least 1 second after the 50th mark
- **AND** the reviewer relaunches `python3 scripts/hitl.py review <drawing-id>`
- **THEN** all 50 marks are reflected in the restored UI state

### Requirement: Performance budget and load-time hard-fail diagnostic

The system SHALL guarantee end-to-end interaction lag of under 50 milliseconds for the marking actions (mark TP, mark FP, delete, add FN, undo, redo, next/previous, zoom in/out) on an A0 drawing at 300 DPI with up to 2000 detection boxes. If the system detects at load time that this budget cannot be met on the current hardware, browser, or drawing, it MUST fail loudly with a single diagnostic message identifying the specific bottleneck (e.g. "DZI tile pyramid missing", "frame-render probe exceeded 50 ms on baseline 2000-box overlay", "tile cache cannot fit minimum working set in 512 MB"). The system MUST NOT silently degrade.

#### Scenario: 50 ms budget is met in the happy path

- **WHEN** the reviewer marks any single detection on an A0/300DPI drawing with up to 2000 boxes
- **THEN** the elapsed time from keystroke to next painted frame is under 50 milliseconds

#### Scenario: Load-time probe fails loudly when budget cannot be met

- **WHEN** the load-time performance probe measures any per-frame work exceeding 50 milliseconds on the baseline overlay
- **THEN** the browser displays a single full-screen diagnostic banner naming the failed probe and the measured value
- **AND** the reviewer cannot begin marking until the underlying cause is addressed

#### Scenario: Missing DZI fails loudly

- **WHEN** the reviewer attempts to open a drawing whose `data/raw/drawings/<id>.dzi` does not exist
- **THEN** the UI refuses to render the drawing and displays a single diagnostic message instructing the reviewer to run `python3 scripts/hitl.py build-tiles <drawing-id>`
- **AND** no single-bitmap fallback is attempted

### Requirement: Forbidden anti-patterns

The system MUST NOT exhibit any of the following anti-patterns, each of which the prior notebook UI carried: modal dialogs in the marking loop, mouse-only workflows for any primary action, full re-renders on every mark, loss of zoom/pan state after a marking action, a separate save step the reviewer must remember, ambiguous visual states where TP and FP look similar, loading the full A0 raster as a single bitmap, fixed hit-test radius that does not adapt to zoom, or any code path that re-exposes the deleted notebook behind a flag.

#### Scenario: No modal in the marking loop

- **WHEN** the reviewer performs any sequence of marking actions
- **THEN** no modal dialog, blocking popup, or focus-stealing prompt appears at any point

#### Scenario: Zoom and pan state survive every action

- **WHEN** the reviewer marks a detection at any non-default zoom and pan position
- **THEN** the viewport zoom and pan position after the action are identical to before the action

#### Scenario: No full re-render on a mark

- **WHEN** the reviewer marks one detection
- **THEN** the rendering work for the action is bounded to the changed detection's overlay region and the progress UI counters
- **AND** the underlying DZI tile layer is not redrawn

### Requirement: Schema-write contract preserves existing corrections table and px_detections.json shape

The system SHALL write FP corrections as rows in the existing `corrections` table of `data/corrections.db` with the existing column layout: `id` (autoincrement primary key), `job_id` (TEXT), `element_type` (TEXT, set to "column"), `element_index` (INTEGER), `original_element` (TEXT containing JSON of the original bbox + score), `changes` (TEXT containing JSON), `is_delete` (INTEGER, set to 1 for FP), `timestamp` (REAL). The system SHALL append FN_ADDED detections to `data/jobs/{job_id}/px_detections.json` under the existing `"columns"` list with the existing record shape `{ "bbox": [x1, y1, x2, y2], "score": 1.0, "source": "human_added" }`. The system MUST NOT modify any existing column of the `corrections` table and MUST NOT change the structural shape of `px_detections.json`. The system SHALL route ALL corrections-DB writes through `_apply_marks` (a single SQLite transaction per batch), and MUST NOT depend on the legacy `scripts/corrections_logger.py` write helpers (`save_job`, `record_delete`, `record_edit`, `record_add`) which were removed in this change because they could not satisfy the batch-as-one-transaction or DELETE_FN-rescind-trap requirements.

#### Scenario: FP write is readable by existing helper

- **WHEN** the UI records an FP mark for `(job_id, element_index)`
- **AND** `scripts/corrections_logger.py::iter_effective_corrections(conn)` is invoked
- **THEN** the FP appears in the iterator's output with `is_delete=1`
- **AND** `scripts/retrain_yolo.py` and `scripts/hard_negative_pool.py` consume the row without any code change

#### Scenario: FN_ADDED write extends px_detections.json without breaking shape

- **WHEN** the UI adds a missed-column detection via drag
- **THEN** `data/jobs/{job_id}/px_detections.json` gains one new entry inside the `"columns"` list with keys `"bbox"`, `"score"`, `"source"` set to `[x1,y1,x2,y2]`, `1.0`, `"human_added"` respectively
- **AND** the `"meta"` subtree is unchanged
- **AND** `scripts/hard_negative_pool.py::_drawing_id_for_job` still resolves the drawing id from the file's meta.source as before

### Requirement: Sidecar tables for UI-only state

The system SHALL persist TP confirmations and reviewer session identity in two NEW SIDECAR tables added to `data/corrections.db` by an idempotent additive migration: `tp_confirmations(session_id TEXT, job_id TEXT, element_index INTEGER, ts REAL, PRIMARY KEY (job_id, element_index))` and `reviewer_sessions(session_id TEXT PRIMARY KEY, reviewer_id TEXT NOT NULL, started_ts REAL NOT NULL)`. The migration MUST NOT alter any existing column, table, index, or trigger.

#### Scenario: Migration is additive only

- **WHEN** the FastAPI backend starts against an existing `data/corrections.db`
- **THEN** the two sidecar tables exist with the columns above
- **AND** the existing `corrections` table's schema (columns, indexes, unique constraints) is byte-identical to its pre-migration state
- **AND** existing rows in `corrections` are not modified

#### Scenario: Existing downstream consumers ignore sidecar tables

- **WHEN** `scripts/retrain_yolo.py` or `scripts/hard_negative_pool.py` reads from `data/corrections.db` after the migration
- **THEN** the consumers ignore the sidecar tables and produce identical output to a pre-migration run with the same correction history

### Requirement: Reviewer identity per session

The system SHALL establish a reviewer identity on first launch by prompting once for a non-empty string and persisting the value at `~/.column-review.json`. On every subsequent launch the system MUST read the persisted reviewer_id and MUST insert a `reviewer_sessions` row with a fresh session_id, the reviewer_id, and the current timestamp. Every `tp_confirmations` row written during the session MUST carry the session's session_id.

#### Scenario: First launch prompts and persists

- **WHEN** the reviewer launches `python3 scripts/hitl.py review <drawing-id>` for the first time on a machine where `~/.column-review.json` does not exist
- **THEN** the browser shows a single non-modal first-launch input asking for a reviewer id
- **AND** on submit the value is written to `~/.column-review.json`
- **AND** a `reviewer_sessions` row is inserted with the new session_id

#### Scenario: Subsequent launches reuse identity

- **WHEN** `~/.column-review.json` already contains `{"reviewer_id": "alice"}`
- **AND** the reviewer launches a review session
- **THEN** no first-launch prompt appears
- **AND** a new `reviewer_sessions` row is inserted with reviewer_id "alice" and a fresh session_id

### Requirement: Single-drawing scope per session

The system SHALL open exactly one drawing per running process, identified by the `<drawing-id>` argument to `python3 scripts/hitl.py review <drawing-id>`. The UI MUST NOT include a drawing switcher, drawing list, or in-session drawing-change action.

#### Scenario: One drawing per process

- **WHEN** the reviewer launches `python3 scripts/hitl.py review TGCH-TD-S-200-L3-00`
- **THEN** the served UI displays only that drawing
- **AND** no in-UI control allows switching to a different drawing without closing the process

#### Scenario: Switching drawings requires relaunch

- **WHEN** the reviewer wishes to review a second drawing
- **THEN** they close the current process and launch `python3 scripts/hitl.py review <other-drawing-id>` separately
