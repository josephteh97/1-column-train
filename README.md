# column-train

A YOLO + YOLO + CNN cascade detector for structural columns in
architectural floor plans, with a single-command web reviewer for
correction and one-button retraining of the two trainable models.

The frozen baseline is a YOLOv11s detector trained on procedurally
generated synthetic floor-plan tiles. The two trainable models — a
yolo11n rescue proposer and a 98 k-param CNN veto — learn from
reviewer corrections logged through the web UI. The baseline is never
fine-tuned, so it cannot regress, while the two trainables specialise
in opposite directions: the rescue YOLO recovers false negatives the
baseline missed, and the CNN classifier rejects false positives from
either detector.

## Architecture

```
PIL.Image (A0 raster)
  → tiled 1280×1280
  → for each tile:
        column_detect.pt   (yolo11s, frozen)     → main proposals
        column_rescue.pt   (yolo11n, trainable)  → rescue proposals
  → union via cross-detector NMS (each survivor tagged detect/rescue/both)
  → post-process pipeline:
        aspect → size → shape → OCR
          → column_classifier.pt (98k CNN, trainable)  ← FP veto
          → centre-NMS → IoU-NMS
  → final detections
```

Three weight files live at the repo root:

| File                       | Params | Role                                  | Retrains |
|----------------------------|--------|---------------------------------------|----------|
| `column_detect.pt`         | ~9 M   | Frozen baseline proposer              | NEVER    |
| `column_rescue.pt`         | ~2.6 M | Trainable proposer (FN recovery)      | ~20 min  |
| `column_classifier.pt`     | ~98 k  | Trainable veto stage (FP rejection)   | ~30 s    |

One UI button — Train Both — retrains both trainables sequentially
in a single click. The CNN classifier finishes first (~30 s, frees
the GPU on exit), then the rescue YOLO runs (~20 min). Failure of
the CNN stage aborts before the rescue stage runs, so the absorption
gate never sees a half-promoted state.

## Install

```bash
# Once per machine. Registers the `column-review` console script.
pip install -e .
```

Dependencies: Ultralytics YOLO11, PyTorch, Pillow, NumPy, OpenCV,
FastAPI + uvicorn, OpenSeadragon (bundled in static assets), Tesseract
(optional, for the OCR filter).

## Quick start

```bash
# 1. Drop a floor plan (PNG / JPG / PDF supported by ingest) into
#    ~/Documents/retrain-dataset/. That's the picker's only source.
cp '/path/to/L3.jpg' ~/Documents/retrain-dataset/

# 2. Launch the reviewer. Picks a free port; opens a browser tab.
column-review
```

In the browser:

1. Pick `L3` from the file picker, type any reviewer id, click Open.
   First click auto-ingests the file (rasterise + DZI tile pyramid,
   ~30–60 s on A0). Re-clicks are instant.
   You can still drive ingest from the CLI with
   `python3 scripts/hitl.py ingest <plan> --drawing-id <id>` for
   scripts or batch workflows.
2. The DZI tile pyramid + the model detections render within ~3 s.
3. Click Run YOLO to see proposals from both detectors. Each
   proposal is tagged in the underlying JSON with `source` =
   `detect` / `rescue` / `both`.
4. Mark corrections:
   - **F** or **click** on a detection toggles it as False Positive.
   - **Left-drag** in empty space adds a missed column (FN_ADDED).
   - **U / Shift-U** undo / redo (≥ 100 levels).
   - **N / P** step to next / previous detection.
   - **J** jump to next unreviewed.
   - **0** = 100 % zoom, **H** = fit, **Space + drag** = pan.
   - Autosave is on. Close the tab anytime.
5. When done with a session, click Train Both. The status pill cycles
   `queued → running → completed`. Both `column_classifier.pt` and
   `column_rescue.pt` update; `column_detect.pt` is untouched.

## Safety: the absorption gate

Clear detections is blocked (HTTP 412) when `corrections.db` holds any
row newer than the latest training cycle for THIS drawing. The gate
reads both `column_classifier.meta.json` and `column_rescue.meta.json`,
takes the minimum per-job `latest_correction_ts_per_job` value, and
refuses Clear if any correction beats it. Recovery: click Train Both.

A missing meta file is treated as never-trained (timestamp `0`), so a
half-deployed system always blocks Clear until the next Train Both
cycle. The structurally safe direction is the conservative one — you
never lose corrections to a Clear that beat the training cycle.

## Inference soft-fail

If either trainable weights file goes missing (rollback, file move,
mid-promotion crash), the cascade prints one stderr diagnostic and
falls back to whatever combination remains valid. Inference never
raises on a missing trainable. This is the rollback path: archive a
.pt file, restart the reviewer, and the corresponding stage drops
out of the pipeline cleanly.

## Command-line equivalents

```bash
# Retrain both (what the Train Both button calls):
python3 scripts/train_both.py

# Retrain only the CNN classifier (FP-veto specialist):
python3 scripts/train_bbox_classifier.py

# Retrain only the rescue YOLO (FN-recovery proposer):
python3 scripts/train_yolo_rescue.py

# Refresh disk pools from data/corrections.db without retraining:
python3 scripts/hard_negative_pool.py            # CNN classifier negatives
python3 scripts/rescue_tile_pool.py              # rescue YOLO tiles + labels

# Status hint based on current correction count:
python3 scripts/hitl.py status
```

## Pointers

- **CLAUDE.md** — Architecture, invariants, training roadmap, and
  design rationale. Deeper than this README; read it before changing
  the cascade.
- **openspec/changes/** — Spec-driven change history. The currently
  archived change `replace-classifier-with-rescue-yolo` records the
  reasoning behind the three-model cascade.
- **data/corrections.db** — SQLite database holding every reviewer
  correction. Source of truth; the two disk pools (`hard_negatives/`,
  `rescue_tiles/`) are derived caches.


## Find-tune Parameters


● Two files, one for each training stage. Here's the map:

  Rescue YOLO (yolo11n) → scripts/train_yolo_rescue.py

  ┌───────────────────────────────────────────────────────────┬───────────────────────┬───────────────┬──────────────────────────────────────────────────────┐
  │                      Hyperparameter                       │         Type          │     Where     │                       Default                        │
  ├───────────────────────────────────────────────────────────┼───────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
  │ --epochs                                                  │ CLI flag              │ line ~177     │ 30                                                   │
  ├───────────────────────────────────────────────────────────┼───────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
  │ --batch                                                   │ CLI flag              │ line ~178     │ 4 (fits 8GB GPU)                                     │
  ├───────────────────────────────────────────────────────────┼───────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
  │ --imgsz                                                   │ CLI flag              │ line ~179     │ 1280 (locked invariant)                              │
  ├───────────────────────────────────────────────────────────┼───────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
  │ --patience                                                │ CLI flag              │ line ~180     │ 10 (early-stop)                                      │
  ├───────────────────────────────────────────────────────────┼───────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
  │ --lr0                                                     │ CLI flag              │ line ~181     │ 5e-4                                                 │
  ├───────────────────────────────────────────────────────────┼───────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
  │ --tau-fn                                                  │ CLI flag              │ line ~184     │ 0.5 (gate FN threshold)                              │
  ├───────────────────────────────────────────────────────────┼───────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
  │ --tau-fp                                                  │ CLI flag              │ line ~185     │ 0.3 (gate FP threshold)                              │
  ├───────────────────────────────────────────────────────────┼───────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
  │ BASE_WEIGHTS                                              │ constant              │ line ~62      │ "yolo11n.pt" — change to yolo11s.pt for bigger model │
  ├───────────────────────────────────────────────────────────┼───────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
  │ Augmentation kwargs (hsv_h, degrees, scale, mosaic, etc.) │ model.train(...) call │ ~line 215-240 │ floor-plan-friendly values                           │
  └───────────────────────────────────────────────────────────┴───────────────────────┴───────────────┴──────────────────────────────────────────────────────┘

  Usage (no file edit needed for CLI flags):
  python3 scripts/train_yolo_rescue.py --epochs 60 --patience 30 --lr0 1e-4 --tau-fp 0.5

  CNN classifier → scripts/train_bbox_classifier.py

  ┌────────────────────┬──────────────────┬────────────────────────────────────────────┬────────────────┐
  │   Hyperparameter   │       Type       │                   Where                    │    Default     │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────┼────────────────┤
  │ --epochs           │ CLI flag         │ line ~539                                  │ 30             │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────┼────────────────┤
  │ --val-frac         │ CLI flag         │ line ~540                                  │ 0.10           │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────┼────────────────┤
  │ --patience         │ CLI flag         │ line ~542                                  │ 5 (early-stop) │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────┼────────────────┤
  │ --seed             │ CLI flag         │ line ~541                                  │ 0              │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────┼────────────────┤
  │ _BATCH_SIZE        │ module constant  │ line ~497                                  │ 64             │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────┼────────────────┤
  │ Learning rate      │ inline in main() │ torch.optim.AdamW(...lr=1e-3...) ~line 670 │ 1e-3           │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────┼────────────────┤
  │ weight_decay       │ inline same line │                                            │ 1e-4           │
  ├────────────────────┼──────────────────┼────────────────────────────────────────────┼────────────────┤
  │ pos_weight formula │ inline           │ n_neg / max(1, n_pos) ~line 661            │ auto-computed  │
  └────────────────────┴──────────────────┴────────────────────────────────────────────┴────────────────┘

  Usage:
  python3 scripts/train_bbox_classifier.py --epochs 50 --patience 15 --val-frac 0.15

  Inference thresholds (after training, affect what the model "sees")

  These don't require retraining — they tune the cascade behavior:

  ┌───────────────────────────────┬───────────────────────────────────────┬────────────────┬────────────────────────────┐
  │             Knob              │                 File                  │    Default     │          Meaning           │
  ├───────────────────────────────┼───────────────────────────────────────┼────────────────┼────────────────────────────┤
  │ --rescue-conf-threshold       │ CLI on column-review                  │ 0.4            │ Rescue YOLO acceptance bar │
  ├───────────────────────────────┼───────────────────────────────────────┼────────────────┼────────────────────────────┤
  │ --classifier-threshold        │ CLI on column-review                  │ 0.5            │ CNN classifier veto bar    │
  ├───────────────────────────────┼───────────────────────────────────────┼────────────────┼────────────────────────────┤
  │ Main YOLO conf_th             │ column_review/inference.py:114        │ 0.25           │ Frozen baseline conf       │
  ├───────────────────────────────┼───────────────────────────────────────┼────────────────┼────────────────────────────┤
  │ MAX_ASPECT, MIN_SIDE_PX, etc. │ scripts/postprocess_pipeline.py:43-47 │ 2.0, 12, 60, … │ Shape filter band          │
  └───────────────────────────────┴───────────────────────────────────────┴────────────────┴────────────────────────────┘

