
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
detections with the review notebook and fold them into the next
fine-tune. The loop closes automatically once corrections are in the
DB.

```
correct_detections.ipynb    →  data/corrections.db + data/jobs/{id}/
scripts/retrain_yolo.py     →  column_detect_ft_{ts}.pt
manual cp                   →  column_detect.pt (deploy)
```

### Steps

1. Open `correct_detections.ipynb` and set `IMAGE_PATH` to the real
   plan you want to correct.
2. Run cells 1–5. Cell 5 registers a new `job_id` and writes
   `data/jobs/{job_id}/render.jpg` + `px_detections.json`.
3. Run cell 6. Page through the thumbnail grid; flip the dropdown
   under each false-positive thumbnail to **DELETE**. Click **Save
   corrections** when done with the whole review.
4. For columns the model missed entirely (no thumbnail to mark), open
   the plan in your image viewer, read off (cx, cy, size_px) for each
   missed column, and add them to the `missed = [...]` list in cell 7.
   Run cell 7 — each entry becomes a `record_add` row in the DB.
5. After accumulating enough corrections across multiple plans
   (≥ 10 by default), run:
   ```bash
   python3 scripts/retrain_yolo.py --epochs 30
   ```
   This builds `data/yolo_finetune/`, fine-tunes from the current
   `column_detect.pt`, and writes `column_detect_ft_{timestamp}.pt`
   at the project root.
6. Inspect the fine-tuned weight on a real plan first. When you're
   satisfied, promote manually:
   ```bash
   cp column_detect_ft_{timestamp}.pt column_detect.pt
   ```

### Schema

The notebook writes through `scripts/corrections_logger.py`, which
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
