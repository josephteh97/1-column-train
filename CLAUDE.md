# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

YOLOv11 single-class detector for structural columns in architectural floor plans.
The model is trained on procedurally generated synthetic floor-plan tiles (not on real plans).
Trained weights `column_detect.pt` at the repo root are the deployed artifact consumed by
downstream tooling.

## Common commands

```bash
# 1. Generate synthetic training data → dataset/column/{images,labels,human_check}/{train,val,test}/
python3 generate_column.py                                  # defaults
python3 generate_column.py --canvases 50                    # 50 A0 canvases instead of NUM_IMAGES default
python3 generate_column.py --clean                          # wipe dataset/column/ first
python3 generate_column.py --no-human-check                 # skip red-bbox overlay tiles
python3 generate_column.py --help                           # full options

# 2. Train from COCO-pretrained yolo11s.pt → produces column_detect.pt
python3 train.py

# 3. If train.py was Ctrl-C'd after mAP converged, pick up best.pt and run eval:
python3 finalize.py

# 4. Gentle fine-tune of existing column_detect.pt on a new dataset
#    (writes column_detect_continued.pt — does NOT overwrite the baseline)
python3 train_continue.py

# 5. Refresh the unified rescue-tile pool from data/corrections.db.
#    Idempotent — run any time after a HITL session to materialise
#    new tiles + labels to data/rescue_tiles/ without retraining.
python3 scripts/rescue_tile_pool.py [--max 2000] [--dry-run]

# 6. Train BOTH the CNN classifier (~30 s) and the rescue YOLO
#    (~20 min) sequentially. column_detect.pt stays frozen. The
#    🧠 Train Both UI button calls this; the absorption gate writes
#    latest_correction_ts_per_job to both meta.json files on pass.
python3 scripts/train_both.py

# 6a. Train only the CNN classifier (the FP-veto specialist).
python3 scripts/train_bbox_classifier.py [--epochs 30]

# 6b. Train only the rescue YOLO (the FN-recovery proposer).
python3 scripts/train_yolo_rescue.py [--epochs 30] [--tau-fn 0.5] [--tau-fp 0.3]

# 7. Launch the web correction reviewer (FastAPI + OpenSeadragon).
#    `pip install -e .` registers the `column-review` console_script;
#    after that the command runs from any directory, auto-picks a
#    free port if the default 8765 is busy, and opens the browser.
#    Requires the DZI tile pyramid: `hitl.py ingest <plan>` builds
#    it inline; for pre-existing drawings use
#    `python3 scripts/hitl.py build-tiles <drawing-id>` first.
column-review
```

There is no test suite, no linter config, and no build step. The deliverable is the `.pt` file.

## Architecture

### Pipeline

```
generate_column.py      → dataset/column/{images,labels,human_check}/{train,val,test}/
  └── single canvas pipeline:
        A0 14043×9929 canvas tiled into 1280×1280 .png patches with rich
        annotation (bubbles, beam labels, dim arrows, north arrows, RC walls,
        internal partitions, detail callouts) plus three failure-mode-focused
        structures: walled openings (X-cross interior), 3-wall stairs with
        zigzag break lines, and chopped-wall lifts with door gaps. Each of
        the three carries dense edge-flanking labelled columns so the model
        learns columns DO sit next to these structures.
        human_check/ holds the same tiles with red bbox overlays for QA.
train.py                → runs/detect/column_detector/weights/best.pt
                          + copies it to ./column_detect.pt
finalize.py             → same as train.py's post-training stage, run standalone
train_continue.py       → reads column_detect.pt, writes column_detect_continued.pt
scripts/rescue_tile_pool.py
                         → reads data/corrections.db + data/jobs/<id>/render.jpg
                         → assembles 1280×1280 tiles + YOLO label files to
                           data/rescue_tiles/{images,labels}/ + manifest.json
                         → idempotent, prunes on rescind, survives ⌫ Clear
scripts/hard_negative_pool.py
                         → reads data/corrections.db is_delete=1 rows
                         → crops 24-px-padded FP regions to
                           data/hard_negatives/*.png + manifest.json
                         → fed to the CNN classifier as the negative class
scripts/train_bbox_classifier.py
                         → reads dataset/column + corrections.db + hard_negatives/
                         → trains the 98k-param CNN classifier on 64×64 crops
                         → writes column_classifier.pt + .meta.json (with
                           latest_correction_ts_per_job for the gate)
                         → column_detect.pt is NEVER touched
scripts/train_yolo_rescue.py
                         → auto-invokes rescue_tile_pool.py at start
                         → trains yolo11n from yolo11n.pt COCO init on
                           synthetic dataset + data/rescue_tiles/
                         → writes column_rescue_quarantine_<ts>.pt, runs
                           scripts/absorption_gate.run_gate against it
                         → on pass: promotes to column_rescue.pt + writes
                           column_rescue.meta.json with
                           latest_correction_ts_per_job map
                         → on fail: quarantine retained, column_rescue.pt
                           untouched, meta.json carries gate_failure block
                         → column_detect.pt is NEVER touched
scripts/train_both.py    → sequential wrapper: CNN classifier then rescue YOLO.
                         → spawned by the 🧠 Train Both UI button.

scripts/ingest_drawings.py → data/raw/drawings/<id>.{png,jpg} + .meta.json
                             + DZI tile pyramid (<id>.dzi + <id>_files/)
column_review/             → FastAPI + OpenSeadragon web reviewer
                             over the DZI tile pyramid. Installed via
                             `pip install -e .` and launched with the
                             top-level `column-review` command from
                             any directory. Writes through
                             corrections_logger into data/corrections.db
                             (existing schema) + sidecar tables
                             (tp_confirmations, reviewer_sessions) +
                             a retrain_jobs tracker. 🧠 Train Both spawns
                             `scripts/train_both.py` as a background
                             subprocess with status polled via
                             `GET /api/jobs/latest`.
```

### Tile-size invariant (critical)

The whole system is calibrated around **TILE_SIZE = 1280 == IMGSZ = 1280**. Synthetic columns
are sized in tile-pixel units (e.g. 16–34 px square, 24–42 px round) to match how columns
appear at inference time when a real A0 plan is tiled with a 1280 window and 200 px overlap.
Changing `IMGSZ` in `train.py` or `TILE_SIZE`/`TILE_STEP` in `generate_column.py` without
changing the other will silently mis-scale columns and regress the baseline.

### Dataset layout convention

`train.py` switches classes via the single constant `CLASS = "column"`. All paths derive from
it: `dataset/<CLASS>/data.yaml`, `<CLASS>_detect.pt`, `runs/detect/<CLASS>_detector/`. To add
door/wall/beam later, drop `dataset/door/` with the same `images/{train,val,test}` +
`labels/{train,val,test}` + `data.yaml` shape and set `CLASS = "door"`.

### Training roadmap (per README + train.py docstring)

- **Phase 1 (current)**: from scratch on columns only → `column_detect.pt`.
- **Phase 2**: load `column_detect.pt`, expand `nc=4`, fine-tune door/wall/stairs with
  `freeze=10` (backbone locked, only upper layers + new head train).
- **Long-term**: single joint `nc=4` run starting from `column_detect.pt`.

### Augmentation policy (intentional)

Floor plans are grayscale axis-aligned line drawings. `train.py` sets `hsv_h=0`, `degrees=0`,
`shear=0`, `perspective=0` deliberately — do not enable rotation/perspective for this class
family. `mosaic=0.5` and `batch=4` together are tuned for 8 GB VRAM; BatchNorm needs ≥4.

### `train_continue.py` safety knobs

`lr0=1e-4`, `epochs=3`, `freeze=15`, `mosaic=0.0` — four guards against catastrophic
forgetting of the existing baseline. Output goes to `column_detect_continued.pt`; promotion
to `column_detect.pt` is **manual** by design (`cp column_detect_continued.pt column_detect.pt`).
Do not auto-overwrite the baseline.

### Architecture C: yolo-yolo-cnn three-model detector

The deployed inference pipeline runs two YOLOs in parallel, unions
their proposals, then puts the result through the CNN classifier
veto stage before final NMS:

```
PIL.Image
  → tiled 1280×1280
  → for each tile:
        column_detect.pt   (frozen yolo11s)   → main_boxes
        column_rescue.pt   (trainable yolo11n) → rescue_boxes
  → union(main_boxes, rescue_boxes) via cross-detector NMS @ IoU=0.15
        (each survivor tagged source = "detect" / "rescue" / "both")
  → run_pipeline:
        aspect / size / shape / [OCR]
          → CNN classifier veto (column_classifier.pt, trainable)
          → centre-NMS / IoU-NMS
  → InferenceResult (boxes + scores + sources + rescue_version + classifier_version)
```

Three trainable artifacts at the repo root, each with a distinct role:

| Artifact | Role | Retrain |
|---|---|---|
| `column_detect.pt` (yolo11s, ~9M params) | Primary proposer — frozen baseline trained on the synthetic dataset. | NEVER — protects against catastrophic forgetting. |
| `column_rescue.pt` (yolo11n, ~2.6M params) | Secondary proposer — recovers FNs the baseline missed. | ~20 min on GPU per cycle. |
| `column_classifier.pt` (98k-param CNN) | Veto stage — rejects FPs from either YOLO. | ~30 s on GPU per cycle. |

One UI button — 🧠 Train Both — retrains BOTH trainables in a single
click. `scripts/train_both.py` runs the CNN classifier first (~30 s,
frees GPU on exit), then the rescue YOLO (~20 min). Sequential by
design: CNN failure aborts before rescue runs so the ⌫ Clear absorption
gate never observes half-promoted state.

```bash
python3 scripts/train_both.py                        # one cycle, both models
python3 scripts/train_bbox_classifier.py             # CNN only
python3 scripts/train_yolo_rescue.py                 # rescue YOLO only
python3 scripts/train_yolo_rescue.py --dry-run       # pool refresh + data.yaml only
```

Inputs (auto-assembled — synthetic dataset is OPTIONAL):
- `dataset/column/{images,labels}/train` — dense synthetic labels, primary
  gradient signal.
- `data/rescue_tiles/{images,labels}` — unified on-disk pool produced by
  `scripts/rescue_tile_pool.py` from `data/corrections.db`:
    - positive tiles: 1280×1280 crops around FN_ADDED locations with EVERY
      accepted positive in the tile labelled (FN_ADDED + tp_confirms +
      un-FP'd model detections — implicit TPs included).
    - negative tiles: 1280×1280 crops around FP locations with an empty
      `.txt` label file. YOLO's standard missing-label supervision teaches
      "no column here". This replaces the previous separate
      `data/hard_negatives/` crop pool.
- Tile content (not source correction) decides the tile's `kind`. An FN
  and FP in the same 1280×1280 area produce a single positive tile with
  the FP simply omitted from the label list.

Output flow:
- `column_rescue_quarantine_<ts>.pt` — every training cycle writes here
  first, NEVER directly to `column_rescue.pt`.
- `scripts/absorption_gate.run_gate` — runs after training, two checks:
  (1) every effective is_delete=0 correction in the latest batch must be
      predicted by the new weights at IoU ≥ τ_fn (default 0.5)
  (2) every effective is_delete=1 correction must have zero predictions
      at IoU ≥ τ_fp (default 0.3)
- On pass: quarantine renames to `column_rescue.pt`,
  `column_rescue.meta.json` records `gate_status="passed"` plus the
  `latest_correction_ts_per_job` map.
- On fail: quarantine retained, `column_rescue.pt` unchanged,
  `column_rescue.meta.json` records `gate_status="failed"` and the
  `gate_failure` block listing the failing correction ids + IoUs.

`column_review/yolo_rescue.py` provides the mtime-keyed loader (same
pattern as `column_review.inference._get_or_load_model` for the main
YOLO) so promoting a fresh `column_rescue.pt` by overwrite auto-
invalidates the inference cache without a server restart. Missing or
unloadable `column_rescue.pt` is a SOFT-FAIL: the cascade prints one
stderr diagnostic and falls back to main-detector-only output. This is
also the rollback path.

### FP/FN absorption safety gate (Architecture C — two-meta check)

`⌫ Clear detections` is blocked (HTTP 412) when `corrections.db` has
any row for `job_id` whose timestamp exceeds the MINIMUM of:
- `column_classifier.meta.json["latest_correction_ts_per_job"][job_id]`
- `column_rescue.meta.json["latest_correction_ts_per_job"][job_id]`

A job is only Clear-safe when BOTH trainables have absorbed every
correction. A missing meta file is treated as `0` (never trained), so
a half-deployed system (one .pt absent) is conservatively blocked
until the next 🧠 Train Both cycle. Recovery is a single click on
🧠 Train Both, which sequentially:
1. Refreshes the hard-negative pool from `data/corrections.db`.
2. Retrains the CNN classifier; writes `column_classifier.pt` +
   `.meta.json` (latest-correction map populated).
3. Refreshes the rescue-tile pool from `data/corrections.db`.
4. Retrains the rescue YOLO; runs the absorption gate; on pass writes
   `column_rescue.pt` + `.meta.json`.

After step 4 succeeds, ⌫ Clear unblocks for the now-current job.

### Why three models (not two)

`column_rescue.pt` alone could in principle handle both jobs (FN
proposal via positive labels, FP rejection via missing-label
supervision at FP locations). Architecture C keeps the CNN as a
separate specialist because:
- The 30-second retrain cadence on the CNN makes per-batch FP
  iteration practical (vs. ~20 min for the rescue cycle).
- The CNN's binary patch-classification objective is more sample-
  efficient than YOLO's tile-level loss for the narrow "is this 64×64
  patch a column?" question.
- Defense in depth: two filter stages with different decision surfaces
  catch failure modes neither would catch alone.

The rescue YOLO is still the primary FN-recovery mechanism — the CNN
cannot propose, only veto. They are complementary, not redundant.

### Model architecture choice

`MODEL_YAML = "yolo11s.pt"` (COCO-pretrained, ~9 M params) is the current default — comments
in `train.py` note that 1300 synthetic tiles is too few to train a non-tiny model from
scratch, and `yolo11m.yaml` regressed on the real plan. `yolo11n-column.yaml` (custom P2
stride-4 head for tiny objects) exists at the repo root but needs ~12 GB VRAM at imgsz=1280
and is reserved for future runs with more data.

## Notes for working here

- The user-facing entry point is `README.md` (project root). Keep its
  workflow narrative in sync with this file when the cascade changes.
- `runs/detect/` accumulates training artifacts — don't delete without checking for the
  current best weights.
- `baseline-pt/column_detect.pt` and `column_detect_prev.pt` are snapshots of past good
  weights; do not overwrite. Promote new weights through the manual copy step in
  `train_continue.py`'s output rather than blowing these away.
