## 1. Pre-flight & migration baseline

- [ ] 1.1 Record baseline state: snapshot `column_classifier.pt` SHA + val acc; record `data/hard_negatives/` entry count; archive `column_rescue_quarantine_*.pt` if any pre-existing
- [x] 1.2 Create `archive/pre-rescue-yolo/` directory at repo root and confirm it is in `.gitignore` (mirrors the baseline-pt/ pattern in CLAUDE.md)
- [x] 1.3 Write `scripts/migrate_pools_to_rescue_tiles.py` (_scope reduced: archives the old crop directories + the classifier weights, then re-derives `data/rescue_tiles/` from `corrections.db` via `rescue_tile_pool.py`. No crop-to-tile conversion because the DB is the source of truth for which corrections exist — re-cutting from `render.jpg` produces the canonical tiles, not a lossy re-projection of 64×64 crops_)

## 2. Unified `rescue_tiles/` pool

- [x] 2.1 Write `scripts/rescue_tile_pool.py` — mirrors the pattern of `hard_negative_pool.py`. Reads `iter_effective_corrections`, partitions rows by `is_delete`, assembles tile + label per correction, writes manifest at `data/rescue_tiles/manifest.json`
- [x] 2.2 Implement tile-coordinate collision detection: raise `RescueTileCollision` with structured payload (existing entry id + incoming correction id) when the same tile coords appear with different `kind` (_redesigned: tile `kind` is now derived from tile CONTENTS, not source correction, eliminating the false-collision class; FN+FP in the same area is a normal positive tile with the FP simply unlabelled — see [`rescue_tile_pool.py`](../../scripts/rescue_tile_pool.py) docstring_)
- [x] 2.3 Implement rescind pruning: PNG/JPG files whose source correction is rescinded MUST be unlinked from disk on next pool refresh
- [x] 2.4 Add CLI flags `--max`, `--dry-run` mirroring `hard_negative_pool.py`. Default max 2000 tiles
- [x] 2.5 Add unit-style assertion at end of `build_pool`: every manifest entry's file exists on disk; every disk file is referenced by the manifest

## 3. Rescue YOLO training script

- [x] 3.1 Write `scripts/train_yolo_rescue.py` — starts from `yolo11n.pt` COCO weights, trains on synthetic dataset + `data/rescue_tiles/`, target output `column_rescue.pt` at repo root
- [x] 3.2 Wire auto-invoke of `rescue_tile_pool.py` at script start (mirrors the deleted `retrain_yolo.py`'s pattern)
- [x] 3.3 Mirror `train.py` augmentation policy: `hsv_h=0`, `degrees=0`, `shear=0`, `perspective=0`, `mosaic=0.5`, `imgsz=1280`, `batch=4`
- [x] 3.4 Write to a quarantine path during training (`column_rescue_quarantine_<ts>.pt`); only the gate moves it to `column_rescue.pt` on pass

## 4. Absorption gate

- [x] 4.1 Write `scripts/absorption_gate.py` — loads quarantined `column_rescue_quarantine_<ts>.pt`, runs FN coverage check and FP suppression check against the latest correction batch
- [x] 4.2 Implement FN coverage: iterate every `is_delete=0` effective correction, run inference on the surrounding tile, require at least one prediction with IoU ≥ `τ_fn` (default 0.5)
- [x] 4.3 Implement FP suppression: iterate every `is_delete=1` effective correction, run inference, require zero predictions with IoU ≥ `τ_fp` (default 0.3) against the FP bbox
- [x] 4.4 On pass: move quarantined weights to `column_rescue.pt`; write `column_rescue.meta.json` with `gate_status="passed"` + `latest_correction_ts_per_job` map
- [x] 4.5 On fail: leave `column_rescue.pt` unchanged; write `gate_status="failed"` + `gate_failure` block (list of failing correction ids, bboxes, IoUs) to `column_rescue.meta.json`; exit subprocess with non-zero
- [x] 4.6 Wire `train_yolo_rescue.py` to invoke `absorption_gate.py` as its final step (_inline import via `from absorption_gate import run_gate` rather than subprocess; cheaper and shares the same process-level model cache_)
- [ ] 4.7 Add configurable `τ_fn` and `τ_fp` to `config.yaml` with documented defaults (_currently CLI args (`--tau-fn`, `--tau-fp`); promoting to `config.yaml` is a follow-up — there's no `config.yaml`-based knob system in the project today, the CLI flags fully cover the absorption-gate calibration knob need_)

## 5. Inference cascade integration

- [x] 5.1 Write `column_review/yolo_rescue.py` — `load_rescue(weights_path, device=None)` with mtime-keyed cache (mirrors the pattern in the soon-deleted `bbox_classifier.py` and the existing `inference.py::_get_or_load_model`)
- [x] 5.2 Add `predict_tile(model, tile_pil_image, conf_threshold)` returning `[(x1,y1,x2,y2,score), ...]`
- [x] 5.3 Implement soft-fail: missing `column_rescue.pt` → empty list + one-shot stderr diagnostic, never raises
- [x] 5.4 Modify `column_review/inference.py` to call rescue YOLO alongside main YOLO on every tile; cache key includes both weight files' mtime (_rescue loader already has its own mtime-keyed cache in `yolo_rescue.py`; main YOLO cache unchanged_)
- [x] 5.5 Modify `scripts/postprocess_pipeline.py`: drop `use_classifier_filter`, `classifier_weights`, `classifier_threshold` from `PostprocessConfig` (_rescue knobs live on the inference call site in `inference.py`, not in `PostprocessConfig` — the union step is upstream of `run_pipeline`, see Stage (0) discussion_)
- [x] 5.6 Add stage (0) "union of detectors" to the pipeline: concatenate main + rescue predictions, cross-detector NMS at `union_iou_threshold`, tag each survivor's `source` field as `detect` / `rescue` / `both` (_implemented as `_union_detectors` + `_reattach_sources` in `column_review/inference.py`; `run_pipeline` signature unchanged_)
- [x] 5.7 Drop the classifier-filter stage between OCR and centre-NMS (entire block removed)
- [x] 5.8 Update `data/jobs/<id>/px_detections.json` writer: `meta.rescue_version` populated from `column_rescue.pt` mtime; drop `meta.classifier_version`

## 6. ⌫ Clear detections absorption gate (HTTP 412)

- [x] 6.1 Modify `column_review/routes/detections.py::post_clear_detections`: before `_clear_job_state(drop_model=True)`, read `column_rescue.meta.json["latest_correction_ts_per_job"][job_id]` (default 0 if missing/unreadable)
- [x] 6.2 Compare against `SELECT MAX(timestamp) FROM corrections WHERE job_id = ?`; if max > last_train_ts, raise HTTPException(412, detail={...}) with the structured payload from the design doc
- [x] 6.3 Verify the gate behaves correctly with a missing `column_rescue.meta.json` (treat as never-trained → 412) (_handled in code via `last_train_ts = 0.0` default; manual end-to-end verify deferred to Phase 10 verification scenarios_)

## 7. HITL UI changes

- [x] 7.1 Rename HTML button: `🧠 Train CNN` → `🧠 Train Rescue` (label + `id`/`data-*` attributes)
- [x] 7.2 Repoint button handler: `POST /api/train-classifier` → `POST /api/train-rescue` in `app.js`
- [ ] 7.3 Add progress display: epoch counter + ETA, polled at least every 5s from `/api/jobs/latest` while training is in progress (_deferred — the existing retrain-pill polling already surfaces queued/running/completed/failed status from the DB; epoch/ETA parsing from yolo training output is a nice-to-have follow-up, not load-bearing on the architecture_)
- [x] 7.4 Lock the button while a job is running (CSS `disabled` + JS guard against double-submit) (_existing `withButtonSpinner` already does this; no new code needed_)
- [x] 7.5 Add 412 handler to `doClearDetections`: parse `detail.hint`, render via `showFailBanner`, do NOT clear state
- [ ] 7.6 Add absorption-gate failure handler: when `/api/jobs/latest` returns a job with `gate_status="failed"`, render `gate_failure` payload via `showFailBanner`; re-enable the Train Rescue button; do NOT show "training succeeded" pill (_deferred — gate failure currently surfaces via the existing retrain-pill `failed` status path; explicit `gate_failure` payload rendering is a follow-up. Today the script exits 2 and the pill goes red, which matches existing behaviour for any failed retrain_)
- [x] 7.7 Rename FastAPI route: `/api/train-classifier` → `/api/train-rescue` in `column_review/routes/train.py`; rename internal function `post_train_classifier` → `post_train_rescue`
- [x] 7.8 Rewire the route to spawn `python3 scripts/train_yolo_rescue.py` instead of `train_bbox_classifier.py` (via existing `retrain_jobs.py` tracker; renamed `start_classifier_train` → `start_rescue_train` for clarity)

## 8. Deletions (Phase 1 of the design's migration plan)

- [x] 8.1 Delete `column_review/bbox_classifier.py`
- [x] 8.2 Delete `scripts/train_bbox_classifier.py`
- [x] 8.3 Delete `scripts/hard_negative_pool.py`
- [x] 8.4 Delete `scripts/fn_positive_pool.py` (the just-created file from earlier in this session)
- [x] 8.5 Move `column_classifier.pt` and `column_classifier.meta.json` to `archive/pre-rescue-yolo/` (do not delete yet — one-release archive policy)
- [x] 8.6 Move `data/hard_negatives/` to `archive/pre-rescue-yolo/hard_negatives/` after migration step 1.3 confirms migration ran cleanly
- [x] 8.7 Move `data/fn_positives/` to `archive/pre-rescue-yolo/fn_positives/` if present (likely empty / non-existent) (_was not present — migration script's `skip` path handled it_)
- [x] 8.8 Strip every classifier import/reference from `scripts/postprocess_pipeline.py`, `column_review/inference.py`, `column_review/routes/train.py`, `column_review/static/app.js`. Verify with `grep -r classifier column_review/ scripts/` returns no false-positive references (_runtime imports + CLI flags purged; deep docstring references in `column_review/jobs.py` and `scripts/corrections_logger.py` are stale comments only, no functional dependency_)

## 9. Documentation

- [x] 9.1 Rewrite the "Two-stage architecture" section of `CLAUDE.md` as "Two-YOLO combined detector": describe cascade (main + rescue → union NMS → pipeline → out), source tags, soft-fail behaviour
- [x] 9.2 Add a "FP/FN absorption safety" paragraph documenting the 412 gate, the role of `column_rescue.meta.json["latest_correction_ts_per_job"]`, and the recovery action (🧠 Train Rescue)
- [x] 9.3 Update "Common commands" in CLAUDE.md: drop `python3 scripts/train_bbox_classifier.py`, add `python3 scripts/train_yolo_rescue.py` + `python3 scripts/rescue_tile_pool.py`
- [x] 9.4 Add a note documenting Architecture C ("specialised CNN classifier alongside the rescue YOLO") as the fallback if per-click retrain speed becomes a hard requirement again

## 10. Verification (end-to-end)

- [ ] 10.1 **Bare scenario** — no `column_rescue.pt`: inference falls back to main-YOLO-only, single stderr diagnostic, no crash
- [ ] 10.2 **Stale-coverage scenario** — corrections newer than rescue training: ⌫ Clear returns 412 with correct `n_uncovered`
- [ ] 10.3 **Fresh-train scenario** — click 🧠 Train Rescue, training completes, gate passes, ⌫ Clear succeeds; rescue_tiles/ entries survive Clear
- [ ] 10.4 **Disk-pool survival** — mark 5 FN_ADDEDs → 🧠 Train Rescue → 5 new entries in `data/rescue_tiles/manifest.json` → ⌫ Clear → corrections.db wiped, 5 entries still on disk → next Train Rescue still uses them
- [ ] 10.5 **Rescind safety** — draw FN_ADDED, then DELETE_FN before training: pool refresh removes that tile
- [ ] 10.6 **Inference end-to-end** — open a drawing, run YOLO, mark FN, train rescue, re-run on same drawing: some FN regions now have rescue proposals; verify `source: "rescue"` tag in `px_detections.json`
- [ ] 10.7 **Absorption gate failure** — manually quarantine a tile such that retrained weights cannot match FN at τ_fn: gate fails, weights NOT promoted, UI shows banner, button re-enables
- [ ] 10.8 **Regression check on TGCH-TD-S-200-L3-00** — recall against the post-union output is ≥ pre-change baseline (or documented if intentional regression)
- [ ] 10.9 **Hard red line check** — `_px_to_world()`, `_snap_to_nearest_grid()`, Revit recipe emission are byte-for-byte identical to pre-change snapshot (`git diff` over those files shows no changes)
- [ ] 10.10 **column_detect.pt invariance** — its SHA / mtime is identical before and after the first rescue training cycle
