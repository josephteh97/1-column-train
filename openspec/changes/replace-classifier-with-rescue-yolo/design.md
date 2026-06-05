## Context

The column-review HITL loop today has three runtime components:

1. `column_detect.pt` (yolo11s, ~9M params, frozen) — primary detector.
   Synthetic-only training. Hard requirement: never regress.
2. Post-process pipeline in `scripts/postprocess_pipeline.py` —
   aspect / size / shape / OCR / centre-NMS filters.
3. `column_classifier.pt` (98k-param CNN, trainable) — patch
   classifier veto stage between OCR and centre-NMS.

The CNN classifier was added to give the HITL loop *some*
trainable component without risking the frozen baseline. It works
for FP rejection (its veto power directly drops bad proposals)
but is structurally incapable of recovering FNs (it has no proposal
mechanism — input is a 64×64 crop of an already-proposed bbox, output
is a single probability). The user's stated objective ("detect FNs
the baseline missed") cannot be reached via this architecture.

This change replaces the CNN with a second, **trainable** YOLO
(`column_rescue.pt`, yolo11n). The frozen baseline stays. Both YOLOs
run in parallel and their outputs are unioned (via NMS) before the
post-process pipeline. The rescue YOLO learns from FN_ADDED bboxes
(positive labels) AND FP regions (label-absent tiles — YOLO's
standard negative supervision). One model, both correction signals,
real proposal capability.

## Goals / Non-Goals

**Goals:**
- Give the HITL loop a trainable component that can RECOVER FNs,
  not just veto FPs.
- Preserve `column_detect.pt` byte-for-byte; never auto-overwrite.
- Unify the two existing correction crop pools
  (`hard_negatives/` + the just-created `fn_positives/`) into one
  on-disk tile pool that YOLO can consume directly.
- Make ⌫ Clear detections impossible to use as a data-loss vector
  by hard-failing (HTTP 412) when corrections haven't been
  absorbed.
- Hard-gate publication of new rescue weights: if the trained model
  doesn't actually learn the latest batch, refuse to publish.

**Non-Goals:**
- Parallel-tile inference of the two YOLOs. Sequential per tile is
  fine until profiling says otherwise.
- A fast-path back to CNN-classifier mode. The classifier is
  archived for one release cycle then deleted.
- A different geometry pipeline. Aspect / size / shape / OCR /
  centre-NMS / IoU-NMS stay byte-for-byte unchanged.
- A different per-class scope. Single-class (`column`) preserved.
- Migration of `corrections.db` rows. Schema unchanged.
- Multi-reviewer / cloud / auth. Out of scope as before.

## Decisions

### D1. Why second YOLO over CenterNet / pretrained-backbone alternatives

Considered alternatives:

| Option | Verdict | Reason |
|---|---|---|
| Add a decoder to the existing CNN | Rejected | 64×64 patch input is the wrong geometry for dense detection; 98k params is undersized; would amount to rebuilding a small detector from scratch. |
| CenterNet + ResNet18 (pretrained on ImageNet) | Considered, rejected | Pretrained features from natural photos don't transfer cleanly to grayscale line drawings. Adds an unproven architecture; the synthetic-data bootstrap gradient would dominate the pretrained signal anyway. ~75% chance of meaningful win. |
| YOLOv11n trained from `yolo11n.pt` COCO init | **Chosen** | YOLO is *the* known-good family for this exact dataset (`column_detect.pt` is existence proof). Single-class small-object detection is YOLO's sweet spot, not a weakness. Same imgsz invariant, same data.yaml format, same augmentation policy. ~85% chance of meaningful win. |
| Bigger CNN (full ResNet18 patch classifier) | Rejected | Doesn't solve the proposal problem — still patch-classifier-shaped. |

Risk delta: YOLOv11n has the narrowest "will this work?" question
because we are inside the architecture family that already works
on the same data.

### D2. Why one rescue YOLO over two trainable models (rescue + classifier)

Considered alternatives:
- **Architecture A** — keep CNN classifier alongside rescue YOLO; both train on
  corrections; redundant veto for defense-in-depth.
- **Architecture B (chosen)** — single trainable component (rescue YOLO);
  YOLO's missing-label supervision at FP locations handles FP rejection.
- **Architecture C** — specialized division of labor: rescue YOLO proposes
  (slow retrain), classifier vetoes (fast retrain).

Architecture B chosen because:
- YOLO standard training already handles FP rejection (missing-label =
  "not a column here"). No need for a second model to add veto power.
- Single learning signal. Both correction streams (FN_ADDED + FP) feed
  one model, one training cycle, one meta.json — simpler absorption gate.
- Removes ~1000 LOC (`bbox_classifier.py`, `train_bbox_classifier.py`,
  `hard_negative_pool.py`, `fn_positive_pool.py`, classifier-filter
  stage).
- Architecture C is the documented fallback if per-click retrain
  speed becomes a hard requirement again. Not a revert path; an
  additive escalation.

### D3. Unified pool storage format

`data/rescue_tiles/images/<drawing>__<hash>.jpg` + matching
`labels/<drawing>__<hash>.txt`. Standard ultralytics YOLO format.
Positive labels (FN_ADDED bboxes + all accepted positives in the
tile) for FN-source corrections; empty `.txt` for FP-source
corrections.

The empty-label-file convention is YOLO's standard "this tile has
no column at this location" signal — no special handling required
in the training loop. This is what eliminates the need for a
separate `hard_negatives/` directory.

Collision policy: if a tile coordinate is already in the pool with
a different encoding (one positive, one empty), the write
hard-fails with a diagnostic naming the conflicting corrections.
No silent overwrite. Rationale: the same tile coordinate appearing
as both positive and negative is a labelling inconsistency the user
needs to resolve, not something the system should paper over.

### D4. Single-class scope preserved

The rescue YOLO emits class id `0` / label `column`, identical to
`column_detect.pt`. Union NMS treats them as the same class. This
preserves the existing single-class invariant in
`bootstrap-yolo-column-system::detection-model::Single-class
YOLOv11s detector`.

### D5. Absorption gate criteria

Two checks, both required to publish:

1. **FN coverage**: for every `is_delete=0` correction in the
   latest batch, the new `column_rescue.pt` must propose at least
   one bbox with IoU ≥ `τ_fn` (default `0.5`) against the
   FN_ADDED bbox.
2. **FP suppression**: for every `is_delete=1` correction in the
   latest batch, the new `column_rescue.pt` must emit zero
   proposals with IoU ≥ `τ_fp` (default `0.3`) against the FP bbox.

Either check failing refuses publication, archives the
quarantined weights to `column_rescue_quarantine_<ts>.pt`, and
surfaces the structured diagnostic (list of failing correction
ids + IoU values + bbox coordinates) to the UI.

The two thresholds are configurable in `config.yaml`. The defaults
match the broader pipeline's existing IoU NMS threshold (0.15)
loosened to 0.3 / tightened to 0.5 respectively to be neither too
strict nor too lenient.

### D6. ⌫ Clear detections gate

Block ⌫ Clear when `corrections.db` has rows newer than
`column_rescue.meta.json["latest_correction_ts_per_job"][job_id]`.
Recovery: click 🧠 Train Rescue. Surfaces as HTTP 412 with a
structured detail body that the frontend renders via
`showFailBanner`.

The gate consults rescue meta.json, not classifier meta.json, since
the classifier is being deleted. A missing meta.json is treated as
"never trained" → `last_train_ts = 0` → every correction is
uncovered → 412.

### D7. Cache invalidation

`column_review/inference.py::_get_or_load_model` uses a
stat-based cache key. Extending the key to include
`column_rescue.pt`'s mtime + size means promoting either model by
overwrite auto-invalidates the cache without a server restart.
Same pattern as the existing classifier cache being deleted.

`scripts/postprocess_pipeline.py`'s `@memory_first` cache key
likewise extends to incorporate `rescue_version` from meta.json.
This is the `meta.json` schema change called out in the proposal.

## Risks / Trade-offs

- **[Rescue YOLO first cycle may regress vs. classifier baseline]** →
  Mitigation: the absorption gate refuses to publish until FN
  coverage hits τ_fn. Worst case the user clicks Train Rescue,
  gets a "gate failed" diagnostic, has to either lower thresholds
  or gather more corrections. No silent regression.
- **[~20-minute retrain breaks the fast-feedback loop]** →
  Trade-off accepted by the user. v5 cadence is "train after a
  batch", not "train after every click". Architecture C remains
  the documented fallback if this becomes painful.
- **[Per-tile inference doubles]** → Two YOLOs sequentially per
  tile. Acceptable on target GPU. If profiling shows it matters,
  parallel inference is a follow-up.
- **[yolo11n undercapacity]** → 2.6M params is genuinely small.
  If post-first-cycle validation shows underfitting, fall back to
  yolo11s for the rescue (4× cost on inference + train). Defer the
  decision to after the first cycle.
- **[Pool collision hard-fail surprises the user mid-session]** →
  Mitigation: the diagnostic names the conflicting correction ids
  + the existing pool entry. The user resolves by un-marking one
  of the conflicting corrections. Not silent.
- **[Migration consolidation may lose data]** → The migration
  script archives both `hard_negatives/` and `fn_positives/` to
  `archive/pre-rescue-yolo/` before deletion. Re-deriving the
  rescue_tiles pool from `corrections.db` (the source of truth)
  is the recovery path if the migration is wrong.

## Migration Plan

1. **Pre-flight** (manual): inspect current
   `data/hard_negatives/` count and current `column_classifier.pt`
   val accuracy. Record as baseline.
2. **Code changes** (single commit landed together):
   - Land all Phase 1 deletions (CNN classifier, both crop pools,
     classifier-filter stage in pipeline).
   - Land all Phase 2 additions (rescue YOLO infrastructure,
     unified tile pool, absorption gate, 412 gate, UI rename).
3. **Data migration** (manual, one-shot):
   - Run `scripts/migrate_pools_to_rescue_tiles.py`. It reads
     `data/hard_negatives/manifest.json` + any
     `data/fn_positives/manifest.json`, resolves each crop's
     surrounding tile from `data/jobs/<id>/render.jpg`,
     assembles the tile + label, writes to
     `data/rescue_tiles/`. Hard-fails on coordinate collisions.
   - Archive `data/hard_negatives/` and `data/fn_positives/` to
     `archive/pre-rescue-yolo/`. Delete the originals.
   - Archive `column_classifier.pt` + `column_classifier.meta.json`
     to `archive/pre-rescue-yolo/`. Delete the originals.
4. **First training cycle** (manual): click 🧠 Train Rescue.
   Watch the absorption gate output. If it passes, the new
   `column_rescue.pt` is published. If it fails, follow the
   structured diagnostic.
5. **Regression check** (manual): run inference on
   `TGCH-TD-S-200-L3-00` (the existing regression benchmark from
   `bootstrap-yolo-column-system::feedback-loop::TGCH-TD-S-200-L3-00
   regression test`). Verify recall vs. the pre-change baseline.

**Rollback**: if the new architecture proves untenable, revert the
landing commit. `archive/pre-rescue-yolo/` contains the classifier
weights and the original pools so the old cascade can be restored
verbatim. After one release cycle without rollback, delete the
archive.

## Open Questions

- **`τ_fn` / `τ_fp` defaults**: 0.5 and 0.3 are placeholders pending
  first-cycle empirical tuning. The thresholds are user-configurable
  from day one so the answer can be measured rather than guessed.
- **yolo11n vs yolo11s capacity for rescue**: deferred to the first
  training cycle. Spec-level requirement is "yolo11n initially;
  may upgrade if first-cycle gate fails on synthetic-data
  regression".
- **Whether to expose two-YOLO union NMS threshold separately from
  the existing IoU NMS at 0.15**: currently spec'd as one shared
  knob. If telemetry shows the union step needs its own threshold,
  add it as a follow-up.
