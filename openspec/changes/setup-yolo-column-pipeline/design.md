## Context

TGCH-style architectural floor plans use a small palette of recognisable
column symbols (filled square, outlined square, filled rect, filled circle,
unshaded circle, etc.) at consistent pixel scales when tiled at A0 / 1280 px.
Real labelled training data does not exist at sufficient volume, so we
generate synthetic A0 canvases procedurally and train YOLOv11 on the tile
output. Real-plan inference is then tiled with the same geometry, with
aggressive post-processing to eliminate the FP classes that survive YOLO's
default NMS.

Current state at the time of this proposal:
- `generate_column.py` produces synthetic tiles with all the column variants,
  structures (openings, stairs, lifts, cores), partitions, annotations, and
  decoy elements, gated against `col_rects` so nothing paints over a column.
- `train.py` (and `train_continue.py`, `finalize.py`) wrap Ultralytics
  YOLO with the architectural-drawing augmentation policy.
- `scripts/postprocess_detections.py` and `test_column.ipynb` cell 5
  implement the six-filter inference pipeline.
- The deployed artifact is `column_detect.pt` at the repo root, promoted
  manually from `runs/detect/<run>/weights/best.pt`.

Stakeholders: model developer (iterating on synthetic data and post-
processing), downstream BIM tooling (consumes `column_detect.pt` and the
post-processing module), QA reviewer (consumes the `human_check/` overlay
tiles).

## Goals / Non-Goals

**Goals:**
- Codify the pipeline that drives **near-zero FPs** on real-plan
  inference while maintaining high recall on every column variant.
- Make the TILE_SIZE / TILE_STEP / column-pixel-size invariant explicit
  and reviewable so it survives future contributors.
- Establish the no-column-blocking contract for synthetic drawers.
- Document the training-time-duplicate / inference-time-NMS labelling
  pattern that is non-obvious but load-bearing.

**Non-Goals:**
- Multi-class (door, wall, stairs as separate classes) — reserved for a
  future `expand-to-nc4` change starting from `column_detect.pt`.
- Online learning from user corrections — `scripts/retrain_yolo.py`
  exists as a placeholder; the corrections-DB pipeline is not in scope.
- A unified UI / GUI wrapper around inference.

## Decisions

### Decision 1: Synthetic data only, no real-plan labels

The model is trained exclusively on procedurally generated A0 canvases. No
manually labelled real plans are required.

**Rationale**: Labelling 500+ columns across many real plans is expensive
and slow; iteration speed on the synthetic generator is the constraint
that dominates time-to-improvement. Real plans are used for QA only, via
the inference path.

**Alternatives considered**:
- Hand-label a corpus of real plans: rejected due to cost and slow
  iteration.
- Semi-supervised (a few labelled real plans + synthetic): may be
  added in a future change, but the synthetic-only baseline is
  sufficient for now.

### Decision 2: Single-class detector (`column`) at YOLO class id 0

All column shape variants (square, rect, circle, combined, unshaded variants)
collapse to YOLO class 0. Internal `cls` ids 0-6 in the generator control
geometry only and are erased before writing the label.

**Rationale**: The downstream consumer needs "find all columns" not
"distinguish column shapes." A single-class detector trains faster, has
fewer FP modes, and is easier to evaluate. The geometry variety in the
generator exists only so the model is robust to every real-plan column style.

**Alternatives considered**:
- Multi-class by shape: rejected — the consumer doesn't care, and
  multi-class training has more failure modes.

### Decision 3: TILE_SIZE = 1280 / TILE_STEP = 1080 calibrated to column pixel sizes

Both the generator and the inference path tile at the same geometry, and
the column-pixel-size constants in the generator are tuned so the trained
model sees columns at the same pixel scale it will see at inference time.

**Rationale**: YOLO is scale-sensitive within the 1-tile imgsz; mismatching
training and inference scales silently halves recall. Locking both to the
same constants is the simplest way to enforce the calibration.

**Alternatives considered**:
- Train with extreme scale augmentation: rejected — adds complexity for
  no benefit when we control the inference path.

### Decision 4: Per-tile centre-in-tile labelling with intentional cross-tile duplicates

A column whose centre falls in the 200-px overlap zone is labelled in
EVERY tile whose extent contains it (2-4 tiles for corner-overlap zones).
Cross-tile dedup is the inference pipeline's responsibility, never the
training pipeline's.

**Rationale**: If we dedup at training time, the same column appears as a
**positive** in one tile and as **background (no label)** in another —
the trainer learns to suppress its own positives, catastrophic for recall.
This decision is recorded as a feedback memory (`feedback_no_train_time_dedup.md`)
because it is counterintuitive and we got it wrong once.

**Alternatives considered**:
- "Closest-tile-centre" owning-tile assignment: tried and reverted —
  caused mass FN on edge-zone columns.
- Non-overlapping tiles at training: rejected — loses edge context.

### Decision 5: Zero-tolerance no-column-blocking contract for late drawers

Every drawer in `generate_image` that paints ink after the column placement
phase (`draw_internal_partitions`, `draw_extra_bubbles`,
`draw_filled_triangle_markers`, `draw_slab_signs`, `draw_small_text_decoys`,
`draw_column_labels`, `draw_empty_intersection_decoys`,
`draw_grid_crossing_decoys`) MUST accept `col_rects` and gate every drawn
element against it.

**Rationale**: A drawer painting ink over a column produces an orphan label
(label points to a now-invisible column body). The orphan scrub is a
safety net but is not 100% reliable — early empirical observations showed
the scrub missing ink-overlap cases (e.g., a filled triangle's dark fill
inside the column bbox looks like a column outline to the scrub). The only
robust contract is "don't paint there in the first place." Codified as the
`feedback_no_column_blocking.md` memory.

### Decision 6: Six-filter inference post-processing pipeline

Filters in order, each tuned to a specific FP class:

| Step | Filter | Eliminates |
|------|--------|------------|
| 0 | Stair-mask via HoughLinesP parallel-line clusters | Stair-step / corner / tread FPs |
| 1 | Aspect ≤ 2.0 | Line-shaped FPs over text strokes |
| 2 | Size ∈ [12, 60] px | Too-small noise, too-large mistakes |
| 3 | Shape: fill ≥ 0.40 OR border ring ≥ 0.35 | Empty seam-ghost bboxes, T-junctions |
| 4 | Centre-distance NMS @ 50 px | Cross-tile seam-ghost duplicates |
| 5 | IoU NMS @ 0.15 | Partial-overlap residuals |

**Rationale**: Each filter operates at the cheapest layer that can
discriminate its FP class. Shape filter (step 3) is the most powerful —
it operates on actual pixel content under the bbox, so empty seam-ghosts
(0% fill, 0% border) and stair T-junctions (low border ratio) are killed
by the same filter for free.

### Decision 7: 1-px LABEL_PAD across YOLO labels, padded rects, and orphan scrub

A single `LABEL_PAD = 1` constant drives the YOLO bbox margin around the
column extent, the placement-spacing buffer in `_padded_rect`, and the
sample inset in `_is_orphan_label`. Changing the constant updates all
three consumers consistently.

**Rationale**: Earlier iterations had `LABEL_PAD = 4` (visible gap around
columns in QA overlays) and a brief `pad=0` (tight, but visually flush
with the outline, which the user wanted to clear). A 1-px margin
visibly clears the outline without inflating labels.

## Risks / Trade-offs

- **[Risk] Synthetic-to-real domain gap** → Mitigation: continuously
  inspect real-plan inference output; failure modes (stair-edge marching
  FPs, seam ghosts, etc.) drive synthetic-data and post-processing fixes
  in the same iteration loop.
- **[Risk] Stair-region detector false-positives drop real columns** →
  Mitigation: `STAIR_REGION_PAD_PX = 10` is tunable; can be reduced to 4-6
  if observed; the default thresholds err on the side of dropping FPs.
- **[Risk] HoughLinesP runtime on full A0 image (5-15 seconds)** →
  Mitigation: acceptable for one-shot inference; downscale 2× if
  batch-processing many plans becomes a use case.
- **[Risk] Re-tuning the post-processing constants for a different
  plan style / DPI** → Mitigation: every constant is at the top of
  `test_column.ipynb` cell 5 and `scripts/postprocess_detections.py`,
  named, and documented with its rationale.
- **[Trade-off] Centre-in-tile duplicates inflate the training-label
  count by ~40% vs. dedup** → Acceptable: training time is dominated by
  GPU forward/backward, not label count.

## Migration Plan

This is the first formal codification of the existing pipeline; no
migration from a prior state is required. Apply-time tasks (see
`tasks.md`) primarily verify that the existing implementation matches
the specs.

Rollback: if a future change inadvertently regresses one of the spec
requirements, the affected memory file (e.g.
`feedback_no_column_blocking.md`) and the corresponding `human_check/`
overlay tile are the authoritative signal that the contract has been
broken.

## Open Questions

- Should `_detect_stair_regions` move from `test_column.ipynb` cell 5
  into `scripts/postprocess_detections.py` so the notebook and the
  importable module share one implementation? (Currently the notebook
  inlines it.)
- Should hard-negative-mining from real-plan inference output become a
  formal workflow? Likely yes after the first real-plan retrain cycle
  to assess whether it's needed.
