## 1. Data pipeline — real ingestion + per-drawing splits + hard-negative pool

- [x] 1.1 Add `scripts/ingest_drawings.py`: rasterise PDF/image input at `INPUT_DPI`, write to `data/raw/drawings/<drawing-id>.png` with a `.meta.json` side-car recording the DPI used
- [x] 1.2 Add `scripts/split_drawings.py`: emit `data/splits/{train,val,test}.txt` using `hash(drawing_id) % 100 → {<70: train, <85: val, else: test}`; deterministic across runs
- [x] 1.3 Add `scripts/hard_negative_pool.py`: read `data/corrections.db` for `is_delete=1` rows, crop the FP region from the matching `data/jobs/{job_id}/render.jpg` with a configurable margin, write to `data/hard_negatives/<drawing-id>__<hash>.png`; maintain a `data/hard_negatives/manifest.json` capped at `MAX_POOL_SIZE` (default 2000)
- [x] 1.4 Verify `generate_column.py` still emits `TILE_SIZE=1280` / `TILE_STEP=1080` and that `_save_tiles` uses the centre-in-tile rule WITHOUT dedup
- [x] 1.5 Verify `LABEL_PAD = 1` is consumed by `_yolo_label`, `_padded_rect`, and `_is_orphan_label` (single source of truth)
- [x] 1.6 Verify bare-stair / bare-lift rate at 50% in `maybe_draw_stair_3wall` and `maybe_draw_lift_chopped`; the else branch calls `_place_unlabelled_corner_negatives`
- [x] 1.7 Add a regression check that asserts 0 column-blocking violations across 50 canvases and 0 orphan labels in the smoke set

## 2. Detection model — OOD hard-fail + configurable knobs + post-processing pipeline

- [x] 2.1 Add `scripts/ood_detector.py` exporting `check_ood(image, dpi: int, raw_tile_detections: list) -> None` that raises `OutOfDistributionError` when DPI ∉ `[210, 420]` or mean per-tile detections ∉ `[0.05, 30]`; both bands are configurable
- [x] 2.2 Wire `check_ood` into the inference path before post-processing; deliver the diagnostic message back to the caller
- [x] 2.3 Refactor the 7-filter post-processing into `scripts/postprocess_pipeline.py` exposing `run_pipeline(img_gray, boxes, scores, *, config) -> (boxes, scores, audit)`; both notebooks import this
- [x] 2.4 Confirm `train.py` sets `amp=False`, `lr0=1e-3`, `imgsz=1280`, `degrees=0`, `shear=0`, `perspective=0`, `hsv_h=0`
- [x] 2.5 Confirm `train.py` copies `runs/detect/<run>/weights/best.pt` to `column_detect.pt` only on a clean exit; `train_continue.py` writes to `column_detect_continued.pt`; promotion is manual
- [x] 2.6 Document the two public inference knobs (`CONF_TH`, `INPUT_DPI`) in `READMD.md`

## 3. Human review interface — idempotent saves + job immutability

- [x] 3.1 Add a `UNIQUE(job_id, element_index, is_delete)` constraint (or equivalent dedup-on-insert) to `data/corrections.db` so re-clicking Save does not write duplicate rows
- [x] 3.2 Add `JobAlreadyCorrected` to `scripts/corrections_logger.py` and raise it from `save_job` when any correction row already exists for that `job_id`
- [x] 3.3 In `correct_detections.ipynb` cell 6, refactor the widget state from outer-scope closures + `page_state` dict into a single `ReviewGrid` class with `job_id` / `marks` as instance attributes
- [x] 3.4 In `correct_detections.ipynb` cell 1, walk up from `Path.cwd()` to find the project root automatically; print a clear error if `scripts/corrections_logger.py` cannot be located
- [x] 3.5 In `correct_detections.ipynb` cell 7 (ADD missed), de-duplicate `record_add` by `(round(cx), round(cy))` so re-running the cell with the same `missed` list does not double-append
- [x] 3.6 Fix `_bbox_has_text` so OCR_MIN_CHARS is required within a SINGLE Tesseract token, not summed across tokens

## 4. Feedback loop — metrics, regression test, manual promotion

- [x] 4.1 In `scripts/retrain_yolo.py`, fix the val-split bug so `n_val = min(n_val, len(job_ids) - 1)` — never empty the train split
- [x] 4.2 In `scripts/retrain_yolo.py`, emit `data/metrics/<revision>.json` after every successful retrain with the schema in `feedback-loop` spec (mAP50, mAP50-95, P, R, fp_rate_per_drawing, revision, n_corrections_consumed, n_hard_negatives, input SHAs)
- [x] 4.3 Add a regression evaluator that runs the model on TGCH-TD-S-200-L3-00 and populates the `regression.tgch_td_s_200_l3_00` sub-object with `expected=440`, `detected=N`, `recall=N/440`
- [x] 4.4 Wire `scripts/hard_negative_pool.py` into `scripts/retrain_yolo.py` so pool entries enter the train split as zero-label images
- [x] 4.5 Confirm `column_detect.pt` is never auto-overwritten by `scripts/retrain_yolo.py`; the output is `column_detect_ft_{timestamp}.pt`

## 5. Final acceptance

- [x] 5.1 `openspec validate bootstrap-yolo-column-system` passes
- [ ] 5.2 End-to-end smoke: ingest one real plan → run inference → review 5 detections → record corrections → retrain on the corrections → confirm `data/metrics/<revision>.json` exists with all required fields (USER ACTION — requires GPU run on real plan)
- [ ] 5.3 Run inference on TGCH-TD-S-200-L3-00; confirm OOD does NOT abort on a properly-rasterised input; record per-drawing FP rate (USER ACTION — requires GPU run)
- [x] 5.4 Append a "Human-in-the-loop correction flow" section to `READMD.md` referencing the ingest, split, pool, review, retrain, promote steps in order
- [ ] 5.5 Archive the prior change with `openspec archive setup-yolo-column-pipeline` (only after this bootstrap change is applied and the new pipeline runs end-to-end) (BLOCKED on 5.2/5.3)
