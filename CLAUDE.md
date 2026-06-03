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

# 5. Fine-tune from user corrections logged in data/corrections.db
python3 scripts/retrain_yolo.py [--epochs 30 --min-corrections 20]

# 6. Launch the web correction reviewer (FastAPI + OpenSeadragon) for a
#    drawing that has already been ingested. Requires the DZI tile pyramid;
#    `hitl.py ingest` builds it inline. For pre-existing drawings use
#    `python3 scripts/hitl.py build-tiles <drawing-id>` first.
python3 scripts/hitl.py review <drawing-id>
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
scripts/retrain_yolo.py → reads data/corrections.db, fine-tunes from corrections

scripts/ingest_drawings.py → data/raw/drawings/<id>.{png,jpg} + .meta.json
                             + DZI tile pyramid (<id>.dzi + <id>_files/)
scripts/correction_app/    → FastAPI + OpenSeadragon web reviewer over
                             the DZI tile pyramid. Launched by `hitl.py
                             review`. Writes through corrections_logger
                             into data/corrections.db (existing schema)
                             + two sidecar tables (tp_confirmations,
                             reviewer_sessions). Replaces the deleted
                             correct_detections.ipynb notebook.
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

### `scripts/retrain_yolo.py` — flywheel (not yet wired up)

Expects `data/corrections.db` (SQLite), `data/jobs/{job_id}/render.jpg`, and
`data/jobs/{job_id}/px_detections.json`. These inputs are not produced inside this repo —
they come from an external corrections logger. The script is in place for when that data
exists; it will fail clean if the DB is missing.

### Model architecture choice

`MODEL_YAML = "yolo11s.pt"` (COCO-pretrained, ~9 M params) is the current default — comments
in `train.py` note that 1300 synthetic tiles is too few to train a non-tiny model from
scratch, and `yolo11m.yaml` regressed on the real plan. `yolo11n-column.yaml` (custom P2
stride-4 head for tiny objects) exists at the repo root but needs ~12 GB VRAM at imgsz=1280
and is reserved for future runs with more data.

## Notes for working here

- The README is misnamed `READMD.md`. It is authoritative for project history; keep changes
  there in sync with `CLAUDE.md`.
- `runs/detect/` accumulates training artifacts — don't delete without checking for the
  current best weights.
- `baseline-pt/column_detect.pt` and `column_detect_prev.pt` are snapshots of past good
  weights; do not overwrite. Promote new weights through the manual copy step in
  `train_continue.py`'s output rather than blowing these away.
