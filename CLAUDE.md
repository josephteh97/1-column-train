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

# 5. Train the CNN classifier from corrections logged in data/corrections.db
#    (column_detect.pt stays frozen; this is the only correction-driven loop)
python3 scripts/train_bbox_classifier.py [--epochs 30]

# 6. Launch the web correction reviewer (FastAPI + OpenSeadragon).
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
scripts/train_bbox_classifier.py
                         → reads data/corrections.db + data/hard_negatives/
                         → trains the second-stage CNN classifier
                         → writes column_classifier.pt at the repo root
                           (auto-promoted; column_detect.pt is NEVER
                            written by the HITL loop)

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
                             a retrain_jobs tracker. 🧠 Train CNN spawns
                             `scripts/train_bbox_classifier.py` as a
                             background subprocess with status polled
                             via `GET /api/jobs/latest`.
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

### Two-stage architecture: YOLO + CNN classifier (recommended path for HITL)

The deployed inference pipeline is now a cascade:

```
PIL.Image
  → YOLO (frozen baseline column_detect.pt)
  → tiled_predict → raw boxes/scores
  → run_pipeline (aspect/size/shape/[OCR]/NMS — existing stages)
  → [optional] CNN classifier_filter on 64×64 crops → filtered boxes/scores
  → InferenceResult
```

YOLO **never gets fine-tuned in this loop** — it stays at the synthetic-baseline
distribution and cannot catastrophically forget. The CNN classifier
(`column_review/bbox_classifier.py`, ~98 k params) is the only learned component
that retrains on reviewer corrections. Train it via:

```bash
python3 scripts/train_bbox_classifier.py                 # full retrain
python3 scripts/train_bbox_classifier.py --dry-run       # dataset audit only
```

Inputs (auto-assembled — synthetic dataset is OPTIONAL, no longer a prereq):
- positives (four sources, any subset is enough to train):
  1. Synthetic — every label box in `dataset/column/labels/train/*.txt`.
     Used if present; absent dataset is logged + skipped. Not regenerated
     as part of Train CNN.
  2. Human-drawn FN_ADDED — every is_delete=0 row in `data/corrections.db`,
     bbox looked up from `data/jobs/<id>/px_detections.json`.
  3. Explicit TP confirmations — every row in `tp_confirmations`.
  4. **Implicit TPs** — every model-source detection in
     `data/jobs/<id>/px_detections.json["columns"]` that is NOT marked as FP
     and NOT `source="human_added"`. Safe for the classifier (unlike YOLO
     retrain) because YOLO stays frozen — worst case the classifier becomes
     permissive; YOLO never regresses. The user's stated workflow ("we
     retrain, keep training") is exactly the iterative refinement story.
- negatives: every PNG in `data/hard_negatives/` (FP crops persisted by
  `scripts/hard_negative_pool.py`).

Class imbalance compensation: the BCE loss uses
`pos_weight = n_neg / n_pos` so the classifier doesn't collapse to
"accept everything" when implicit TPs vastly outnumber the curated FPs.

Output: `column_classifier.pt` + `column_classifier.meta.json` at the repo root.
The model cache in `column_review/bbox_classifier.py` is keyed on
`(path, mtime, size)`, so overwriting the .pt invalidates the cache without
a server restart — same pattern as `_get_or_load_model` for YOLO weights.

Pipeline injection: `scripts/postprocess_pipeline.py::PostprocessConfig` has
`use_classifier_filter`, `classifier_weights`, `classifier_threshold`. The
classifier stage sits AFTER OCR and BEFORE centre-NMS so duplicate FPs of the
same wrong thing don't both survive. Soft-fails when the .pt is missing — the
pipeline still produces the YOLO-only output, so the cascade is opt-in by
deployment.

Why this beats fine-tuning YOLO on corrections: the column-review workflow
treats every un-clicked detection as an implicit TP. With one-drawing
corrections, that fed grid bubbles + dim text into the training set as
"positive columns" and the fine-tuned model regressed to detecting *those*
instead of structural columns. The classifier only trains on **explicit**
labels (FP click = "not column", FN_ADDED draw = "column") so the noisy-label
failure mode is eliminated structurally.

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
