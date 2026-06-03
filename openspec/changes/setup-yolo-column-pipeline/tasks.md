## 1. Synthetic data generation — codify and verify

- [ ] 1.1 Confirm `TILE_SIZE = 1280` and `TILE_STEP = 1080` constants in `generate_column.py` are at module top with a comment marking them as the calibration invariant
- [ ] 1.2 Confirm `LABEL_PAD = 1` is the single source of truth used by `_yolo_label`, `_padded_rect`, and `_is_orphan_label`
- [ ] 1.3 Confirm every late drawer (`draw_internal_partitions`, `draw_extra_bubbles`, `draw_filled_triangle_markers`, `draw_slab_signs`, `draw_small_text_decoys`, `draw_column_labels`, `draw_grid_crossing_decoys`) accepts `col_rects` and gates every drawn element via `_bbox_overlaps_any`
- [ ] 1.4 Confirm `_save_tiles` uses the centre-in-tile rule WITHOUT dedup — no `_owning_tile` or closest-tile-centre logic
- [ ] 1.5 Confirm `maybe_draw_stair_3wall` and `maybe_draw_lift_chopped` each have the `random.random() >= 0.30` bare-variant gate
- [ ] 1.6 Add a smoke-test target (e.g. `python3 generate_column.py --clean --out /tmp/colsmoke --canvases 2`) that completes in <60s and produces `/tmp/colsmoke/{images,labels,human_check}/`
- [ ] 1.7 Add a no-block regression check (the `/tmp/check_no_block.py` pattern) that asserts 0 blocking violations across 50 canvases
- [ ] 1.8 Add an orphan-scrub regression check that asserts 0 orphan labels across the smoke set

## 2. Model training — codify and verify

- [ ] 2.1 Confirm `train.py` uses `imgsz=1280`, `degrees=0`, `shear=0`, `perspective=0`, `hsv_h=0`, `batch=4`
- [ ] 2.2 Confirm `train.py` copies `runs/detect/<run>/weights/best.pt` to `column_detect.pt` at the repo root only on a clean training exit
- [ ] 2.3 Confirm `train_continue.py` writes to `column_detect_continued.pt` and prints a manual-`cp` instruction; never auto-overwrites `column_detect.pt`
- [ ] 2.4 Confirm `finalize.py` recovers `best.pt` after a Ctrl-C'd run and updates `column_detect.pt`
- [ ] 2.5 Document the training command in the README / CLAUDE.md (one-liner: `python3 train.py`)

## 3. Inference post-processing — codify and verify

- [ ] 3.1 Confirm `test_column.ipynb` cell 5 implements the six-filter pipeline in order: stair-mask, aspect, size, shape, centre-NMS, IoU-NMS
- [ ] 3.2 Confirm cell 5 short-circuits with `boxes_final = np.zeros((0, 4))` when `all_boxes` is empty
- [ ] 3.3 Confirm `_detect_stair_regions` gracefully returns `[]` when `cv2` is unavailable
- [ ] 3.4 Confirm `scripts/postprocess_detections.py` exposes `filter_detections`, `aspect_filter`, `center_distance_nms`, `cross_tile_nms` and passes its self-test (run `python3 scripts/postprocess_detections.py`)
- [ ] 3.5 Verify the per-stage drop counts print in cell 5 so a user can trace which filter dropped each FP
- [ ] 3.6 Run end-to-end on the reference real plan (`TGCH floor plan/L5.jpg` or equivalent) and capture the count at each stage in the notebook output

## 4. Documentation and memory

- [ ] 4.1 Confirm `CLAUDE.md` references the calibration invariant (TILE_SIZE / TILE_STEP / column-pixel-sizes)
- [ ] 4.2 Confirm `feedback_geometry_table_format.md` memory exists and is current
- [ ] 4.3 Confirm `feedback_no_train_time_dedup.md` memory exists and is current
- [ ] 4.4 Confirm `feedback_no_column_blocking.md` memory exists and is current
- [ ] 4.5 Update `READMD.md` (the misnamed project README) to reference this openspec change once apply is complete

## 5. Final acceptance

- [ ] 5.1 Run `openspec validate --change setup-yolo-column-pipeline` and pass
- [ ] 5.2 Generate a fresh dataset, retrain, and inspect ~10 `human_check/` tiles — confirm no column blocking, no missing labels, tight bbox margin
- [ ] 5.3 Run inference on the reference real plan, confirm FPs in the inspection histogram are ≤ user-stated acceptance threshold
- [ ] 5.4 Promote the new `column_detect.pt` (manual `cp`) and archive this change via `/opsx:archive`
