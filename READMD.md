# Workflow at a glance

Three independent loops use the same artefacts. Pick the one that matches your situation.

```
┌────────────────────────────────────────────────────────────────────────┐
│ A. COLD START — build the deployed model from scratch (synthetic only)│
└────────────────────────────────────────────────────────────────────────┘
   python3 generate_column.py --clean       # 1. regen dataset/column/
   python3 train.py                         # 2. trains → column_detect.pt
   (optional) python3 finalize.py           #    if you Ctrl-C'd after mAP plateau

┌────────────────────────────────────────────────────────────────────────┐
│ B. INSPECT — sanity-check the deployed weight on a real plan          │
└────────────────────────────────────────────────────────────────────────┘
   Open test_column.ipynb in Jupyter, set IMAGE_PATH at the top, run all cells.
   Outputs an annotated PNG under output/. No corrections recorded.

┌────────────────────────────────────────────────────────────────────────┐
│ C. HOT LOOP — improve the model from reviewer corrections (HITL)      │
│    ONE command per phase via scripts/hitl.py                          │
└────────────────────────────────────────────────────────────────────────┘
   1. PREP     python3 scripts/hitl.py ingest <plan> --drawing-id <id>
                  → rasterises + refreshes splits + tells you what to do next

   2. REVIEW   python3 scripts/hitl.py review <drawing-id>
                  → launches the local web reviewer in the browser
                  → press T for TP, F for FP, D to clear a mark, A to add
                    a missed column (drag), U/Y for undo/redo
                  → autosave is on; close the browser when done

      (any time)  python3 scripts/hitl.py status
                  → how many corrections have I accumulated?

   3. RETRAIN  python3 scripts/hitl.py retrain [--epochs 30]
                  → refreshes the FP→hard-neg pool, runs the fine-tune,
                    prints the metrics file + the manual-cp line.
                  Then inspect data/metrics/<ts>.json AND
                  re-run loop B with WEIGHTS=column_detect_ft_*.pt.
                  Promote manually:
                      cp column_detect_ft_<ts>.pt column_detect.pt
```

## Worked example — reviewing TGCH-TD-S-200-L3-00 (L3.jpg)

Concrete copy-paste for the plan at `/home/jiezhi/Documents/TGCH floor plan/L3.jpg`.
**Each command is one line** — don't split them with backslashes (a stray space
after `\ ` turns into a literal-space argument and confuses argparse).

```bash
# Phase 1 — PREP. Quote the path because it contains a space.
python3 scripts/hitl.py ingest '/home/jiezhi/Documents/TGCH floor plan/L3.jpg' --drawing-id TGCH-TD-S-200-L3-00
```

Then launch the web reviewer for that drawing-id:

```bash
# Phase 2 — REVIEW. Browser opens to a local FastAPI viewer with the
# detections overlaid. T = TP, F = FP, D = clear, A = drag-add a missed
# column. Autosave is on; close the browser when done.
python3 scripts/hitl.py review TGCH-TD-S-200-L3-00
```

```bash
# Anytime — check how many corrections you've accumulated.
python3 scripts/hitl.py status

# Phase 3 — RETRAIN. Defaults are fine; bump --epochs if you have lots of corrections.
python3 scripts/hitl.py retrain --epochs 30

# After inspecting data/metrics/<ts>.json and the new weight on a real plan:
cp column_detect_ft_<ts>.pt column_detect.pt
```

### What each placeholder means

| Placeholder | What to put | Example |
|---|---|---|
| `<plan>` | Path to the PDF or image you're reviewing. **Quote it if the path has spaces.** | `'/home/jiezhi/Documents/TGCH floor plan/L3.jpg'` |
| `--drawing-id <id>` | Stable identifier for this drawing. Reusing the same id on the same plan groups its corrections. Use kebab-case. | `TGCH-TD-S-200-L3-00` |
| `--epochs N` | How many epochs to fine-tune. Default 30 works for a first retrain; bump to 50+ if you have >100 corrections. | `--epochs 30` |
| `--dry-run` | (flag, no value) Build `data/yolo_finetune/` only; skip the GPU training step. Sanity-check before committing GPU time. | `--dry-run` |
| `<ts>` (in the cp line) | The timestamp the retrain script printed after it wrote `column_detect_ft_*.pt`. Tab-complete in the shell or just `ls column_detect_ft_*.pt`. | `column_detect_ft_1717369200.pt` |

## When to run which file

| You want to… | Run this | Produces |
|---|---|---|
| **HITL: ingest a real plan + prep splits** | `python3 scripts/hitl.py ingest <plan> --drawing-id <id>` | `data/raw/drawings/<id>.png` + `data/splits/*.txt` |
| **HITL: check how many corrections you have** | `python3 scripts/hitl.py status` | terminal output; effective counts (rescinded auto-filtered) |
| **HITL: refresh pool + retrain + show next step** | `python3 scripts/hitl.py retrain [--epochs 30]` | `column_detect_ft_{ts}.pt` + `data/metrics/<ts>.json` |
| Regenerate synthetic training data | `python3 generate_column.py [--clean] [--canvases N]` | `dataset/column/{images,labels,human_check}/` |
| Train from scratch | `python3 train.py` | `runs/detect/column_detector/weights/best.pt` → copied to `column_detect.pt` |
| Recover after Ctrl-C training | `python3 finalize.py` | Copies the latest `best.pt` to `column_detect.pt` |
| Gentle fine-tune on a new dataset | `python3 train_continue.py` | `column_detect_continued.pt` (manual `cp` to promote) |
| Inspect current weight on a real plan | open `test_column.ipynb` | `output/<plan>_columns.png` |
| Mark FPs / missed columns on a real plan | `python3 scripts/hitl.py review <drawing-id>` | rows in `data/corrections.db`, files under `data/jobs/{job_id}/` |
| Ingest a real plan (PDF/image) at calibrated DPI | `python3 scripts/ingest_drawings.py <src> --drawing-id <id>` | `data/raw/drawings/<id>.png` + `.meta.json` + `.dzi` tile pyramid (~25-35% extra disk) |
| (Re)build the DZI tile pyramid for an already-ingested drawing | `python3 scripts/hitl.py build-tiles <drawing-id>` | `data/raw/drawings/<id>.dzi` + `<id>_files/` tile JPEGs |
| Refresh per-drawing splits | `python3 scripts/split_drawings.py` | `data/splits/{train,val,test}.txt` |
| Build the FP → hard-negative training pool | `python3 scripts/hard_negative_pool.py` | `data/hard_negatives/<id>__<hash>.png` + `manifest.json` |
| Fine-tune from accumulated corrections | `python3 scripts/retrain_yolo.py --epochs 30` | `column_detect_ft_{ts}.pt` + `data/metrics/<ts>.json` |
| Smoke-test synthetic generator | `python3 scripts/check_regression.py --canvases 2` | `OK — no orphan labels.` (or first offending tiles) |

## Quick mental model

- **Synthetic data + train.py** is the COLD path: how the deployed model is built when there is nothing else.
- **`hitl.py review` (web reviewer) + retrain_yolo.py** is the HOT loop: how the deployed model gets better as reviewers find errors on real plans. The web reviewer (FastAPI + OpenSeadragon over a DZI tile pyramid) replaces the old `correct_detections.ipynb` notebook entirely. Corrections are persisted in `data/corrections.db`; rescinded deletes (delete then later edit on the same detection) are automatically filtered out at every read site (`build_dataset`, `hard_negative_pool`, `summary()`).
- **test_column.ipynb** is read-only QA: never writes corrections, never trains, never promotes weights.
- **`column_detect.pt` is never auto-overwritten.** Promotion is always a manual `cp`. The retrain script writes `column_detect_ft_{ts}.pt`; you decide whether to deploy it after inspecting `data/metrics/<ts>.json` and re-running `test_column.ipynb` on a real plan.

---

  What changed 2026.02.25                                                                                                                                                                                                  
                                                                                                                                                                                                                 
  1. Bounding box annotations (_yolo_label)                                                                                                                                                                      
  - Already correct YOLO format; added 4 px padding around every bbox so the model sees a small context border around each column.                                                                               
  - New DRAW_DEBUG_BOXES = False flag at the top — set it True to render red bounding boxes + "column" text directly on the images for quick visual QA.                                                          
                                                                                                                                                                                                                 
  2. Train / Val / Test split                                                                                                                                                                                    
  - Directories are now dataset/images/train/, dataset/images/val/, dataset/images/test/ (matching labels/ dirs created automatically).                                                                          
  - Default ratio: 70 % train · 20 % val · 10 % test — configurable via TRAIN_RATIO / VAL_RATIO at the top of the file.                                                                                          
  - data.yaml now correctly points to each split folder instead of the old flat images/ path.

  3. Revit-style balloon labels                                                                                                                                                                                  
  - Radius scaled to the image: IMG_WIDTH // 90 → IMG_WIDTH // 60 (~45–68 px on a 4096 px canvas), up from the previous 20–32 px which was invisible at any normal zoom.                                         
  - Bold TrueType font loaded from the system (DejaVuSans-Bold.ttf resolves on this machine); falls back gracefully on older Pillow.                                                                             
  - Text centred using anchor="mm" (Pillow ≥ 8) with a textbbox fallback for older builds.                                                                                                                       
  - White-filled circle — the bubble covers the dashed line end, exactly as Revit renders it.                                                                                                                    
  - Grid lines now run bubble-centre → bubble-centre so they terminate cleanly at the annotation bubbles.    

What was built                                                                                                                                                                                                 
                                                                                                                                                                                                               
  yolo11n-column.yaml — custom architecture, no download needed                                                                                                                                                  
  - Adds a P2 (stride-4) detection head to the standard YOLOv11 FPN/PAN                                                                                                                                          
  - Columns at 5–13 px at imgsz=1280 are at the edge of P3's range; P2 gives the model a proper stride-4 feature map to anchor on                                                                                
  - 2.67 M parameters (nano scale) — fast to train on this dataset                                                                                                                                               
  
   What's new in train.py                                                                                                                                                                                         
                                                                                                                                                                                                                 
  plot_training_results() — runs after training, reads results.csv and produces learning_curves.png:                                                                                                             
  - Row 1: Box loss / Class loss / DFL loss (train vs val) + mAP50 vs mAP50-95
  - Row 2: Precision, Recall, mAP50, mAP50-95 each as individual curves with the best epoch marked

  evaluate() — called twice with plots=True:
  - Once on val split → saves to runs/column_detector/eval_val/
  - Once on test split → saves to runs/column_detector/eval_test/
  - Each produces its own confusion matrix, PR curve, F1 curve, P curve, R curve

                                                                                                                                                                                                                 
  train.py — supervised, from scratch                                                                                                                                                                            
  - Loads yolo11n-column.yaml (no .pt download)                                                                                                                                                                  
  - After training: copies best weights → column_detect.pt automatically                                                                                                                                         
  - Runs a test-split evaluation and prints mAP / precision / recall                                                                                                                                             
                                                                                                                                                                                                                 
  ---                                                                                                                                                                                                            
  On your learning-approach question                                                                                                                                                                             
                                                                                                                                                                                                                 
  ┌────────────────────────┬───────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐                                     
  │        Approach        │     Suitable?     │                                                            Why                                                            │                                     
  ├────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                     
  │ Supervised (YOLO)      │ ✅ Use this       │ Labeled synthetic data → direct regression to boxes+classes                                                               │                                     
  ├────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                     
  │ Reinforcement learning │ ❌ Wrong paradigm │ RL optimises sequential policies via reward signals — not designed for detection, vastly more complex for the same result │                                     
  ├────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                     
  │ Transfer learning      │ ✅ Phase 2+       │ Load column_detect.pt, fine-tune for door/wall/stairs — backbone already knows floor-plan features                        │                                     
  ├────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Multi-task (long-term) │ ✅ Best end state │ One nc=4 model trained jointly on all classes — single inference call, shared backbone                                    │                                     
  └────────────────────────┴───────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘                                     
                                                                                                                                                                                                                 
  Phase 2 snippet (when you have door/wall/stairs data):                                                                                                                                                         
  model = YOLO("column_detect.pt")   # backbone carries over                                                                                                                                                     
  model.train(data="dataset_all.yaml", epochs=50, freeze=10)                                                                                                                                                     
  # freeze=10 locks the first 10 backbone layers; only upper features                                                                                                                                            
  # and the new 4-class Detect head are updated initially                                                                                                                                                        
                                                                                                                                                                                                                 
  Run training now:                                                                                                                                                                                              
  python3 train.py                               
  
  
  
  ● Clean. Here's the final structure:                                                                                                                                                                             
                                                                                                                                                                                                                 
  generate-yolo-training-dataset-columns/                                                                                                                                                                        
  ├── generate_column.py       ← generate synthetic images + labels                                                                                                                                              
  ├── train.py                 ← train the model                                                                                                                                                                 
  ├── yolo11n-column.yaml      ← model architecture (P2 head, from scratch)                                                                                                                                      
  ├── READMD.md                ← your notes (kept)                                                                                                                                                               
  └── dataset/                                                                                                                                                                                                   
      ├── data.yaml            ← dataset config (paths + class names, needed by train.py)                                                                                                                        
      ├── images/                                                                                                                                                                                                
      │   ├── train/  (140)                                                                                                                                                                                      
      │   ├── val/    ( 40)
      │   └── test/   ( 20)
      └── labels/
          ├── train/  (140)
          ├── val/    ( 40)
          └── test/   ( 20)



● All paths resolve correctly. Here's the final layout:
                                                                                                                                                                                                                 
  dataset/                                                                                                                                                                                                       
  └── column/                 ← generate_column.py writes here                                                                                                                                                   
      ├── data.yaml                                                                                                                                                                                              
      ├── images/                                                                                                                                                                                                
      │   ├── train/  (560)                                                                                                                                                                                      
      │   ├── val/    (160)                                                                                                                                                                                      
      │   └── test/   ( 80)                                                                                                                                                                                      
      └── labels/                                                                                                                                                                                                
          ├── train/  (560)                                                                                                                                                                                      
          ├── val/    (160)                                                                                                                                                                                      
          └── test/   ( 80)                                                                                                                                                                                      
                                                                                                                                                                                                                 
  To add a new class later, you just:                                                                                                                                                                            
  1. Drop (or generate) a folder dataset/door/ with the same structure + its own data.yaml                                                                                                                       
  2. Change one line in train.py:                                                                                                                                                                                
  CLASS = "door"   # was "column"                                                                                                                                                                                
                                                                                                                                                                                                                 
  That's it — weights, run folder, and data path all update automatically from that single variable.  


## Human-in-the-loop correction flow

When `column_detect.pt` is wrong on a real plan, mark the bad / missing
detections in the local web reviewer (`hitl.py review <drawing-id>`)
and fold them into the next fine-tune. The loop closes automatically
once corrections are in the DB.

```
hitl.py review <drawing-id>  →  data/corrections.db + data/jobs/{id}/
scripts/retrain_yolo.py      →  column_detect_ft_{ts}.pt
manual cp                    →  column_detect.pt (deploy)
```

### Steps

1. Ingest the plan once (also builds the DZI tile pyramid for the
   reviewer; ~25-35 % extra disk):
   ```bash
   python3 scripts/hitl.py ingest <plan> --drawing-id <id>
   ```
2. Launch the web reviewer for that drawing-id:
   ```bash
   python3 scripts/hitl.py review <id>
   ```
   The default browser opens to a FastAPI + OpenSeadragon viewer.
   First launch prompts once for a reviewer-id (stored in
   `~/.column-review.json`).
3. Mark detections with the keyboard. **T** = TP, **F** = FP, **D** =
   clear a mark, **A** = enter add-mode then drag to record a missed
   column, **U** / **Y** = undo / redo, **N** / **P** = next / previous
   unreviewed, **0** = 100 % zoom, **F** = fit, **Space-drag** = pan,
   **Shift-drag** = rubber-band-select then plain release = batch FP,
   Ctrl release = batch delete. Autosave is on — every mark is durable
   to disk within 1 s.
4. After accumulating enough corrections across multiple plans
   (≥ 10 by default), run:
   ```bash
   python3 scripts/retrain_yolo.py --epochs 30
   ```
   This builds `data/yolo_finetune/`, fine-tunes from the current
   `column_detect.pt`, and writes `column_detect_ft_{timestamp}.pt`
   at the project root.
5. Inspect the fine-tuned weight on a real plan first. When you're
   satisfied, promote manually:
   ```bash
   cp column_detect_ft_{timestamp}.pt column_detect.pt
   ```

### Schema

The web reviewer writes through `scripts/corrections_logger.py`, which
maintains the schema `scripts/retrain_yolo.py` expects:

| File | Contents |
|------|----------|
| `data/corrections.db` | SQLite. Table `corrections(id, job_id, element_type, element_index, original_element JSON, changes JSON, is_delete, timestamp)`. One row per correction. |
| `data/jobs/{job_id}/render.jpg` | The plan as reviewed. |
| `data/jobs/{job_id}/px_detections.json` | `{ "columns": [{"bbox": [x1,y1,x2,y2], "score": float, ...}, ...] }`. ADDed entries are appended here at review time. |

`element_index` indexes into `px_detections.json["columns"]`. For
DELETE rows, the retrain skips that index when generating labels. For
ADD rows, the new entry is already in the list and the retrain emits
it as a positive label. EDIT-bbox rows mutate the entry's bbox in
`px_detections.json` AND log the original for audit.

The retrain script is in `scripts/retrain_yolo.py`; it preserves the
fine-tune in `runs/detect/correction_feedback/`. The deployed weight
is **never** auto-overwritten — promotion is a manual `cp` step.

## Inference configuration — two public knobs

The deployed detector exposes exactly two inference knobs; everything
else is deterministic given the loaded weight + these two values.

| Knob | Default | What it controls |
|------|---------|------------------|
| `CONF_TH` | `0.25` | Confidence threshold — detections with `conf < CONF_TH` are dropped before post-processing. |
| `INPUT_DPI` | `300` | The DPI at which a real plan is rasterised before tiling. Tiling geometry (`TILE_SIZE=1280`, `TILE_STEP=1080`) was calibrated at this DPI. |

Both live at the top of `test_column.ipynb` and are surfaced as CLI
flags on `python3 scripts/hitl.py review` (`--conf-th`, `--input-dpi`
in future versions; currently inference is run upstream by the user
before review).

### Out-of-distribution hard failure

Inference aborts with `OutOfDistributionError` instead of emitting
low-quality predictions when:
- **Effective DPI ratio** falls outside `[0.7, 1.4]` (defaults
  `INPUT_DPI / TRAINING_DPI=300`), OR
- **Mean per-tile raw detection count** falls outside `[0.05, 30]`.

See `scripts/ood_detector.py`. The bands are configurable per
deployment; the defaults reject 150-DPI scans and blank pages.
