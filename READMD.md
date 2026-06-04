# Workflow at a glance

Three independent loops use the same artefacts. Each block is a copy-pasteable
shell session — the loop-C example uses `/home/jiezhi/Documents/TGCH floor plan/L3.jpg`
(drawing-id `TGCH-TD-S-200-L3-00`); substitute your own paths.

```bash
# A. COLD START — build column_detect.pt from synthetic data only.
python3 generate_column.py --clean                 # regen dataset/column/
python3 train.py                                   # trains → column_detect.pt
python3 finalize.py                                # optional, if you Ctrl-C'd after mAP plateau
```

```bash
# B. INSPECT — sanity-check column_detect.pt on a real plan.
#   Open test_column.ipynb in Jupyter, set IMAGE_PATH at the top of
#   the notebook, then run all cells. Outputs an annotated PNG under
#   output/<plan>_columns.png. No corrections are recorded.
```

```bash
# C. HOT LOOP — improve column_detect.pt from reviewer corrections.
#   One command per phase via scripts/hitl.py.

# 0. one-time: install web-reviewer runtime deps.
pip install fastapi uvicorn

# 1. ingest — rasterises the source + builds DZI tile pyramid + refreshes splits.
#    (Quote the source path because it contains a space.)
python3 scripts/hitl.py ingest \
    '/home/jiezhi/Documents/TGCH floor plan/L3.jpg' \
    --drawing-id TGCH-TD-S-200-L3-00

# 2. review — launches the local FastAPI app, auto-opens browser at
#    http://127.0.0.1:8765/. On first open the progress strip shows a
#    green "Run inference" button — click it (CPU ~30-90 s, GPU ~2-5 s)
#    to populate the detection overlay with column_detect.pt's predictions.
#    Then mark each box with:
#      T = TP (true positive)        F = FP (drop at retrain)
#      D = clear the mark            A + drag = add a missed column
#      U / Y = undo / redo           N / P = next / previous unreviewed
#      Shift+drag = rubber-band-select (release → batch FP, Ctrl-release → batch delete)
#    Autosave is on; close the browser tab when done.
python3 scripts/hitl.py review TGCH-TD-S-200-L3-00

# (any time) check effective correction counts.
python3 scripts/hitl.py status

# 3. retrain — refreshes the FP→hard-neg pool, fine-tunes from
#    column_detect.pt, writes column_detect_ft_<ts>.pt + data/metrics/<ts>.json.
python3 scripts/hitl.py retrain --epochs 30

# 4. promote — manual, after inspecting data/metrics/<ts>.json AND
#    eyeballing the new weight in test_column.ipynb (loop B).
cp column_detect_ft_<ts>.pt column_detect.pt
```

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

  What changed 2026.06.03 — `rebuild-correction-ui-web` (OpenSpec change)

  1. Reviewer UI: deleted `correct_detections.ipynb`. Replaced by a
     local FastAPI + OpenSeadragon web reviewer launched via
     `python3 scripts/hitl.py review <drawing-id>`. Keyboard-first
     (T/F/D/A/U/Y/N/P/0/F/Space/Shift), tile-pyramid pan/zoom
     (deep-zoom DZI) so A0-at-300DPI plans open in under 3 s,
     undo/redo ≥100 levels, rubber-band batch-mark-FP / batch-delete,
     mini-map with unreviewed-cluster highlights, autosave-per-action
     (≤1 s durable). Load-time perf probe fails loudly if the
     `<50 ms` interaction budget can't be met — no silent degrade.

  2. Tile pyramid: `scripts/ingest_drawings.py` now emits a DZI
     tile pyramid alongside the canonical raster (tile 256, JPEG q80,
     overlap 1). Adds ~25–35 % disk per drawing. Use
     `python3 scripts/hitl.py build-tiles <drawing-id>` to backfill
     drawings ingested before this change. `--no-tiles` opts out of
     pyramid generation; the reviewer then refuses to open the
     drawing with a clear diagnostic.

  3. Schema (additive only): two new sidecar tables in
     `data/corrections.db` — `tp_confirmations` for TP marks and
     `reviewer_sessions` for reviewer-id provenance. Existing
     `corrections` table columns + `data/jobs/<id>/px_detections.json`
     shape are unchanged. `scripts/retrain_yolo.py` and
     `scripts/hard_negative_pool.py` continue to read the original
     schema unchanged.

  4. Storage layer (`scripts/corrections_logger.py`): the legacy
     write helpers (`save_job`, `record_delete`, `record_edit`,
     `record_add`, `JobAlreadyCorrected`) were removed — the web
     reviewer inlines its writes into one SQLite transaction per
     batch via `_apply_marks` in `correction_app/app.py`, which the
     old per-call connections could not deliver. Public read surface
     (`new_job_id`, `iter_effective_corrections`, `summary`, the path
     constants) is preserved verbatim.

  5. New runtime deps: `pip install fastapi uvicorn`. Existing
     dependencies otherwise unchanged.

  6. Removed dead files: `correct_detections.ipynb`,
     `scripts/postprocess_detections.py` (superseded by
     `scripts/postprocess_pipeline.py`), `yolo26n.pt` (orphan
     weight), and the OSD navigation-button image set under
     `scripts/correction_app/static/vendor/images/` (the reviewer
     disables OSD nav controls in favour of keyboard).

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

### Where the web reviewer lives

The reviewer is a **local FastAPI app**, not a hosted service. Source:

```
scripts/correction_app/
├── __init__.py
├── app.py                       ← backend: endpoints, sidecar tables,
│                                  job lookup, single-transaction
│                                  mark writer (`_apply_marks`)
└── static/
    ├── index.html               ← UI shell
    ├── styles.css               ← four-state palette + layout
    ├── app.js                   ← OSD viewer, overlay canvas,
    │                              keyboard, undo/redo, batch ops,
    │                              mini-map, perf probe
    └── vendor/
        └── openseadragon.min.js ← vendored OSD 4.1.0
```

### Launching the web reviewer

```bash
# One-time install of the runtime deps for the web app:
pip install fastapi uvicorn

# Then for any ingested drawing:
python3 scripts/hitl.py review <drawing-id>
```

What the command does, in order:

1. **Imports check** — verifies `fastapi` + `uvicorn` are importable.
   Missing? It prints the `pip install` line above and exits non-zero.
2. **Drawing + DZI check** — resolves `<drawing-id>` via
   `resolve_drawing` and refuses to start if the DZI tile pyramid is
   missing, printing exactly which `build-tiles` command to run.
3. **Sidecar migration** — runs `CREATE TABLE IF NOT EXISTS` for
   `tp_confirmations` and `reviewer_sessions`. Idempotent; safe on
   every launch.
4. **Job bootstrap** — looks up an existing job for this drawing-id +
   raster mtime; if none, creates a fresh one. `render.jpg` encoding
   runs on a daemon thread so the foreground returns immediately
   (under 3 s open requirement).
5. **Port pick** — starts at `127.0.0.1:8765` and walks up to +20
   ports until it finds a free one. Configurable with `--port`.
6. **Browser open** — `webbrowser.open(...)` ~1.5 s after uvicorn
   starts. Terminal prints `Serving correction reviewer at
   http://127.0.0.1:<port>/` so you can paste the URL manually if
   your default browser doesn't auto-open.
7. **First launch** — a non-modal bar at the top of the UI prompts
   once for a reviewer-id. On submit it persists to
   `~/.column-review.json` (atomic write); every subsequent launch
   reuses it. Marking is blocked (server returns 409) until this is
   set — orphan rows in `tp_confirmations` are forbidden by design.

CLI flags on `hitl.py review`:

| Flag | Default | What |
|---|---|---|
| `--port N` | 8765 | Starting TCP port (loopback). Next free port wins. |
| `--tile-cache-mb N` | 512 | Browser-side OSD tile cache ceiling. LRU-evicted past this. |
| `--hit-tolerance-px N` | 8 | CSS-pixel hit-test radius at 100 % zoom; scales up at lower zoom. Pass `0` for pixel-perfect. |
| `--snap-grid-px N` | 0 | If > 0, FN-add bboxes snap to this raster-pixel grid on commit. |

To stop the reviewer: Ctrl-C in the terminal. The browser tab can be
closed at any time — autosave means there is no unsaved state.

### Failure modes (intentional — no silent degrade)

The reviewer **fails loud** rather than degrading silently. You'll see
a single full-screen banner with one of:

- **"DZI tile pyramid missing on disk"** — run
  `python3 scripts/hitl.py build-tiles <drawing-id>` and reopen.
- **"Performance probe exceeded the 50 ms budget"** — the synthetic
  2000-box overlay render-once test on this machine exceeded the
  interaction-lag SLA. Try a smaller drawing or a faster machine; no
  fallback is offered.
- **`409 reviewer_id is not set`** — submit the reviewer-id prompt
  bar at the top of the page first. The bar re-appears automatically
  on the first marking attempt if needed.

### Steps

1. Ingest the plan once (also builds the DZI tile pyramid for the
   reviewer; ~25-35 % extra disk):
   ```bash
   python3 scripts/hitl.py ingest <plan> --drawing-id <id>
   ```
2. Launch the web reviewer for that drawing-id (see "Launching the
   web reviewer" above for what happens behind the scenes):
   ```bash
   python3 scripts/hitl.py review <id>
   ```
   The default browser opens to a FastAPI + OpenSeadragon viewer.
   First launch prompts once for a reviewer-id (stored in
   `~/.column-review.json`).
3. Mark detections with the keyboard. The full shortcut set:

   | Key | Action |
   |---|---|
   | **T** | mark active detection as TP (true positive — correct call) |
   | **F** | mark active detection as FP (false positive — drop at retrain) |
   | **D** | clear / undo the mark on the active detection (FP → unreviewed; TP → unreviewed; FN_ADDED → remove). REMOVED slots are hidden from the overlay; press U to bring one back. |
   | **A** | enter add-mode; the next mouse-drag commits a new FN_ADDED bbox |
   | **U** / **Y** | undo / redo (≥100 levels, O(1) per action) |
   | **N** / **P** | jump viewport to next / previous unreviewed detection |
   | **+** / **-** | zoom in / out centred on cursor or selection |
   | **0** | 100 % (1:1 pixel) zoom |
   | **F** (no active selection) | fit-to-window |
   | **Space-drag** | pan |
   | **Shift-drag** | rubber-band select; plain release → batch-mark-FP; Ctrl-release → batch-delete |
   | **Esc** | clear selection + leave add-mode |
   | click on zoom indicator | type an exact percent and press Enter |

   Mouse-wheel zoom is anchored on the cursor. Autosave is on — every
   mark is durable to disk within 1 s and survives Ctrl-C / browser
   close / kill -9. There is no "Save" button by design.
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


   ### Runtime files
   ```bash
   *.dzi, 
   *.meta.json, 
   hard_negatives/manifest.json
   test.txt
   train.txt
   val.txt
   .jpg

### Schema

The web reviewer writes directly to the on-disk shape consumed by
`scripts/retrain_yolo.py` and `scripts/hard_negative_pool.py`. All
mark writes funnel through a single SQLite transaction per batch
(`scripts/correction_app/app.py::_apply_marks`) — JSON file first
via `os.replace`, then `conn.commit()` — so a crash between the two
can never leave the DB pointing at a JSON entry that doesn't exist.

| Path | Contents |
|------|----------|
| `data/raw/drawings/<drawing-id>.png` (or `.jpg`) | Canonical raster, written by `hitl.py ingest`. |
| `data/raw/drawings/<drawing-id>.meta.json` | `{ drawing_id, dpi, source, ingested_as, size, ingested_ts, dzi_path }`. |
| `data/raw/drawings/<drawing-id>.dzi` + `<drawing-id>_files/<L>/<col>_<row>.jpg` | Microsoft DZI tile pyramid (tile 256 px, overlap 1 px, JPEG quality 80). Built inline by `hitl.py ingest`; backfill via `hitl.py build-tiles`. Adds ~25–35 % disk on top of the raster. |
| `data/corrections.db` table `corrections(id, job_id, element_type, element_index, original_element JSON, changes JSON, is_delete, timestamp)` | One row per mark. **FP** → `is_delete=1`, `changes={}`. **FN_ADDED** → `is_delete=0`, `changes={"bbox":…,"source":"human_added"}`. **DELETE_FN** → `is_delete=1`, `changes={"action":"delete_fn"}` (the state-map distinguisher; the prior `is_delete=0` add row is removed in the same transaction so `iter_effective_corrections` can't rescind the delete). **RESCIND_FP** → `is_delete=0` row that the rescind-on-read invariant uses to hide the prior `is_delete=1`. **RESTORE_FN** → undoes a DELETE_FN by dropping the `is_delete=1` audit row and re-inserting the original `is_delete=0`. |
| `data/corrections.db` table `tp_confirmations(session_id, job_id, element_index, ts)` | **New sidecar**. One row per **TP** mark. Additive only; downstream readers ignore it. Cleared via **CLEAR_TP**. |
| `data/corrections.db` table `reviewer_sessions(session_id, reviewer_id, started_ts)` | **New sidecar**. One row per session start. The reviewer-id is prompted once on first launch and persisted to `~/.column-review.json`. |
| `data/jobs/<job_id>/render.jpg` | The plan as reviewed; written by a daemon thread so it doesn't block first-load. Consumed by `hard_negative_pool.py` for FP crops at retrain time. |
| `data/jobs/<job_id>/px_detections.json` | `{ columns: [{bbox, score, source?}, …], meta: { source, created_ts, raster_mtime, n } }`. `source: "human_added"` flags FN_ADDED entries. `raster_mtime` is checked at reopen so a re-ingested drawing doesn't inherit a stale render.jpg. |

`element_index` indexes into `px_detections.json["columns"]`. For
`is_delete=1` rows, retrain skips that index when generating labels.
For human-added (`is_delete=0` + `source:human_added`) rows, retrain
emits the entry as a positive label. The rescind invariant
(`iter_effective_corrections` in `corrections_logger.py`) is the
single source of truth for "which corrections are live" and is the
shared read path for `retrain_yolo`, `hard_negative_pool`, and the
reviewer's own state map.

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

Both live at the top of `test_column.ipynb`. The web reviewer
itself does NOT run inference — it consumes a pre-populated
`data/jobs/<job_id>/px_detections.json`. If no detections file
exists for the drawing, the reviewer bootstraps an empty job and you
can drag-add FN_ADDED entries by hand; for a real review against the
deployed model, run `test_column.ipynb` (or the upstream inference
script of your choice) first and place its output at
`data/jobs/<job_id>/px_detections.json` before launching
`hitl.py review`.

### Out-of-distribution hard failure

Inference aborts with `OutOfDistributionError` instead of emitting
low-quality predictions when:
- **Effective DPI ratio** falls outside `[0.7, 1.4]` (defaults
  `INPUT_DPI / TRAINING_DPI=300`), OR
- **Mean per-tile raw detection count** falls outside `[0.05, 30]`.

See `scripts/ood_detector.py`. The bands are configurable per
deployment; the defaults reject 150-DPI scans and blank pages.
