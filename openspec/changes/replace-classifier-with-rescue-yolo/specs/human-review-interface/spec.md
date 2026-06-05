## ADDED Requirements

### Requirement: 🧠 Train Rescue retrain control

The HITL UI SHALL expose a single retrain control labelled
"🧠 Train Rescue" that, on click, invokes `POST /api/train-rescue`,
which spawns `python3 scripts/train_yolo_rescue.py` as a tracked
subprocess. The control SHALL be disabled while a training job is
in progress for any drawing.

The endpoint SHALL be `POST /api/train-rescue` (the previous
`/api/train-classifier` endpoint is removed; see REMOVED
Requirements). No legacy alias SHALL be served.

#### Scenario: Successful Train Rescue click
- **WHEN** the user clicks the 🧠 Train Rescue button
- **AND** no other training job is currently running
- **THEN** `POST /api/train-rescue` returns a job id
- **AND** `scripts/train_yolo_rescue.py` is running as a tracked
  subprocess
- **AND** the button is disabled until the subprocess exits

#### Scenario: Train Rescue click while training is running
- **WHEN** a training job is already in progress
- **AND** the user attempts to click 🧠 Train Rescue
- **THEN** the control is disabled (visually + functionally)
- **AND** no second training subprocess is started

#### Scenario: No legacy classifier endpoint
- **WHEN** a client sends `POST /api/train-classifier`
- **THEN** the response is HTTP 404 — no compatibility alias
  forwards to the rescue endpoint

### Requirement: Retrain progress display

The HITL UI SHALL display rescue training progress in two pieces
of information visible without opening developer tools:

1. An **epoch counter** in the form `N / total`, updated at least
   once per epoch.
2. An **ETA** in the form `~M min remaining`, derived from
   elapsed-per-epoch × (total - N).

The fields SHALL come from `column_rescue.meta.json` (partially
written during training) and / or `data/jobs/<latest>/training.log`
parsed by the polling endpoint `/api/jobs/latest`. The frontend
polls at least once every 5 seconds while a job is in progress.

Because rescue training takes ~20 minutes on the target GPU, the
progress display is required — a silently busy button is not
acceptable for that duration.

#### Scenario: Progress visible mid-training
- **WHEN** rescue training is at epoch 12 of 50 and the user opens
  the UI
- **THEN** the UI displays "Epoch 12/50" (or equivalent) and an
  ETA line within 5 seconds of the next poll tick

#### Scenario: Progress cleared on completion
- **WHEN** training subprocess exits with code 0 and the
  absorption gate publishes new weights
- **THEN** the progress indicator clears within 5 seconds
- **AND** the button label returns to enabled "🧠 Train Rescue"

### Requirement: Surface absorption-gate failures to the reviewer

When `scripts/train_yolo_rescue.py` exits with the absorption gate
failing (FN coverage or FP suppression below threshold), the HITL
UI SHALL surface the structured diagnostic from
`column_rescue.meta.json["gate_failure"]` via the existing
`showFailBanner` mechanism. The banner SHALL include:

- A human-readable summary (`N FN(s) not absorbed; M FP(s) still
  firing`).
- A link or details-expand for the per-correction list (correction
  id, bbox, IoU).
- A pointer to retry after gathering more corrections OR adjusting
  `τ_fn` / `τ_fp` in `config.yaml`.

The UI SHALL NOT advance to a "training succeeded" state when the
gate fails. The 🧠 Train Rescue button SHALL re-enable so the user
can retry after addressing the diagnostic.

#### Scenario: Gate failure banner displayed
- **WHEN** the rescue training subprocess exits non-zero with
  `gate_failure` written to meta.json
- **AND** the frontend polls `/api/jobs/latest`
- **THEN** the response includes the gate_failure payload
- **AND** the UI renders the banner via `showFailBanner`
- **AND** the UI does NOT show the "training succeeded" status pill

#### Scenario: Train Rescue button re-enabled after gate failure
- **WHEN** the gate failure banner is shown
- **THEN** the 🧠 Train Rescue button is enabled (so the user can
  retry after gathering more corrections)

### Requirement: ⌫ Clear detections respects the absorption gate

The HITL UI SHALL handle HTTP 412 from `POST /api/detections/clear`
by rendering the response body's `detail.hint` via
`showFailBanner` rather than silently failing or showing a raw
error. The 🧠 Train Rescue button SHALL be visible as the
recovery action.

#### Scenario: Clear blocked surfaces correct guidance
- **WHEN** the user clicks ⌫ Clear detections on a drawing whose
  latest corrections postdate the last rescue training
- **THEN** `POST /api/detections/clear` returns 412 with the
  structured detail body
- **AND** the UI renders `detail.hint` (e.g., "4 correction(s) on
  this drawing have not been included in any rescue-YOLO training
  yet. Click 🧠 Train Rescue to absorb them; then Clear is safe.")
- **AND** the job state is unchanged

## REMOVED Requirements

### Requirement: 🧠 Train CNN retrain control

**Reason:** Superseded by 🧠 Train Rescue (see above). The CNN
classifier (`column_classifier.pt`) is removed from the cascade
entirely; its dedicated retrain control is replaced by the
rescue-YOLO control. The button label, endpoint, and underlying
subprocess all change.

**Migration:** `column_review/static/app.js` is updated in this
change to call `POST /api/train-rescue` instead of
`/api/train-classifier`. The button's `id` and `data-*` attributes
that explicitly reference "classifier" are renamed to "rescue" in
the same change so the JS / CSS bindings stay consistent. No
backwards-compatibility alias is served — the old endpoint
returns 404.

### Requirement: Fast (~30 s) per-click retrain feedback loop

**Reason:** The classifier's ~30-second retrain cycle was an
artifact of its 98k parameter count and 64×64 crop input. The
rescue YOLO (yolo11n, ~2.6M params, full-tile training) takes
~20 minutes per cycle on the target GPU. The HITL workflow shifts
from per-click iteration to per-batch (one training cycle per
HITL session of multiple drawings). The progress-display
requirement above covers the longer wait.

**Migration:** No data-side migration. UX: the reviewer now
clicks 🧠 Train Rescue at the END of a session (or after
gathering enough corrections to make a cycle worth running)
rather than after every individual correction. If the per-click
loop becomes a hard requirement in the future, "Architecture C"
(specialised CNN classifier alongside the rescue YOLO) is the
documented fallback — not a revert to the current state.
