## Context

TGCH-style 2D architectural and structural floor plans contain
hundreds of concrete columns per drawing that drive downstream
quantity, BIM, and review workflows. The repository today contains a
synthetic-only YOLOv11 column detector, a tiled-inference notebook
with a 7-filter post-processing pipeline, and a first-cut HITL review
notebook + corrections-DB logger. The prior change
`setup-yolo-column-pipeline` codified the synthetic-only pipeline.

The system at hand is not yet end-to-end auditable: there is no
real-data ingestion path, no per-drawing split discipline, no formal
out-of-distribution failure behaviour, no hard-negative pool, and no
per-revision metrics record. Reviewers can mark corrections but the
loop back into training has only a manual retrain script, not a
documented contract for what FPs / FNs / edits become at retrain time.

Stakeholders: the model developer (iterating on data + model + post-
processing), the structural / BIM reviewer (consumes the deployed
weight + the review interface), the downstream Revit tooling (consumes
`column_detect.pt`), and the auditor (consumes
`data/metrics/<revision>.json`).

Constraints carried forward from the prior change:
- Single-class scope; class id `0` / string `column`.
- `TILE_SIZE = 1280`, `TILE_STEP = 1080` calibrated to column pixel
  size.
- Manual promotion of `column_detect.pt`; no auto-overwrite.

## Goals / Non-Goals

**Goals:**

- Specify a closed feedback loop from real-plan inference through
  human review back into training, with per-revision metrics so each
  retrain is auditable.
- Make the real-data and synthetic-data paths cohabit one pipeline,
  with stratified per-drawing splits that never leak boxes across
  splits.
- Establish OOD hard-failure as a first-class signal — the deployed
  model abstains on inputs it cannot reliably score.
- Name TGCH-TD-S-200-L3-00 (440 instances) as one regression test so
  every retrain is gauged against a known floor.

**Non-Goals:**

- Column type / family / dimension classification (handled
  downstream).
- Multi-class extension (door / wall / slab / beam).
- Modifications to the existing v3/v4 Revit C# add-in.
- Online learning that bypasses the corrections loop.
- Auto-promotion of fine-tuned weights to `column_detect.pt`.
- Re-tuning the stair-mask post-processing filter (kept optional).

## Decisions

### Decision 1: Supersede the prior change rather than extend it

`setup-yolo-column-pipeline` codified what already existed for the
synthetic pipeline. This change absorbs those three capabilities
(`synthetic-data-generation`, `model-training`,
`inference-post-processing`) into four new capabilities that fold the
post-inference pipeline into `detection-model` and add real-data
ingestion + HITL + feedback loop.

**Rationale**: The user asked for all four parts "specified together"
as one auditable artefact. Extension would leave readers chasing
requirements across two specs.

**Alternatives considered**:
- Extend the prior change with delta `MODIFIED Requirements`.
  Rejected: the prior change's capability decomposition does not map
  cleanly to the new system parts.

### Decision 2: Synthetic + real data cohabit one pipeline

Both real (PDF/image ingestion at configurable DPI) and synthetic
(generated from `generate_column.py`) feed into the same per-drawing
split manifest. Synthetic tiles are treated as one large synthetic
"drawing" for split purposes; real drawings each get their own
drawing ID.

**Rationale**: Real-plan iteration is slow and label cost is high;
synthetic generation closes the labelled-data gap. The model learns
the joint distribution, and the per-drawing split discipline applies
uniformly.

**Alternatives considered**:
- Two separate trainers (real-only + synthetic-only). Rejected:
  doubles the train time and prevents the model from learning the
  joint distribution that real-plan inference actually exhibits.

### Decision 3: Per-drawing stratified splits, deterministic by hash

`hash(drawing_id) % 100 < 70` → train; `< 85` → val; else test. The
hash is stable across runs and adding a new drawing only ever moves
its own boxes into one split (never moves existing drawings).

**Rationale**: Boxes from the same drawing share lighting, ink
density, label fonts, and grid spacing — leaking them across splits
inflates val metrics and hides regressions. Per-drawing
deterministic-hash splits remove that risk and require no central
catalog.

**Alternatives considered**:
- Random shuffle. Rejected — not reproducible.
- Manual JSON manifest. Rejected — requires upkeep on every new
  ingestion.

### Decision 4: Hard-negative pool as the FP → training contract

FPs caught by HITL become background tiles in
`data/hard_negatives/`, included in the next train split as
zero-label images. This raises the model's background score on the
exact patterns the reviewer rejected.

**Rationale**: Re-training without explicit negatives means the model
can re-learn the same FP from synthetic noise. The pool gives FPs a
durable, auditable presence in training.

**Alternatives considered**:
- Add FP regions to a synthetic decoy generator. Rejected: indirect,
  loses the FP's exact pixel evidence, and requires the synthetic
  pipeline to model every FP pattern.
- Weight FPs in the loss function. Rejected: ultralytics does not
  expose per-sample weighting on the standard training path.

### Decision 5: OOD hard-failure via two cheap signals

Inference aborts with `OutOfDistributionError` when (a) effective
DPI is outside `[210, 420]` (= 0.7×–1.4× of training DPI) or (b) the
mean per-tile raw detection count is outside `[0.05, 30]`. Both signals
are cheap to compute and reject both wildly-mis-scaled inputs and
blank pages.

**Rationale**: A silent fallback that emits low-confidence noise on
OOD input is worse than no output — the reviewer trusts the deployed
model and pushes garbage into the corrections loop. Hard failure
forces the operator to fix the rasterisation or re-tile.

**Alternatives considered**:
- Train an explicit OOD classifier. Rejected: extra model, extra
  training data, more failure modes.
- Energy-score / Mahalanobis on penultimate features. Rejected:
  ultralytics doesn't expose the penultimate cleanly; complexity not
  justified for v1.

### Decision 6: HITL persistence via SQLite + per-job directory

Corrections live in `data/corrections.db` (single table) and per-job
artefacts under `data/jobs/{job_id}/`. Schema matches what
`scripts/retrain_yolo.py` already reads. The review interface in v1
is a Jupyter notebook with ipywidgets paginated thumbnails; v2 may
ship a standalone application but the schema does not change.

**Rationale**: SQLite handles the row-append workload, the schema is
already consumed by the existing retrain script (no translation
layer), and ipywidgets ships with Jupyter so no new deployment story
is required. The schema is the load-bearing contract; the UI is the
variable.

**Alternatives considered**:
- Flat JSON of corrections per job. Rejected — querying across jobs
  becomes manual.
- Streamlit / Gradio app. Reserved for v2; out of scope for the
  bootstrap.

### Decision 7: Metrics persistence as per-revision JSON files

Each retrain writes `data/metrics/<revision>.json` with the metrics
schema specified in `feedback-loop`. The revision is the timestamp of
the retrain (or, if running inside a git tree, the SHA). Files are
append-only; older files are never rewritten.

**Rationale**: Independent files per revision give the auditor an
immutable trail without needing a metrics DB. The JSON is
human-readable and trivially diffable across revisions.

**Alternatives considered**:
- Extend `corrections.db` with a `metrics` table. Rejected: mixes
  two unrelated concerns and complicates the corrections schema that
  retrain_yolo.py expects.
- TensorBoard logs. Kept as a parallel emission for live monitoring,
  but the per-revision JSON is the auditable record.

## Risks / Trade-offs

- **[Risk] Real-plan label cost is high** → Mitigation: synthetic
  data does the heavy lifting; real labels are used as fine-tune +
  regression. The HITL loop converts inference errors into labels
  cheaply, so the real-label corpus grows organically.
- **[Risk] OOD band `[210, 420]` may reject legitimate inputs** →
  Mitigation: bands are configurable per deployment. If a downstream
  site uses 200 DPI scans, the band moves; the spec defines the
  defaults, not the only allowed values.
- **[Risk] Hard-negative pool can grow unbounded** → Mitigation:
  retain only the most-recent `MAX_POOL_SIZE` (default 2000) entries,
  weighted by per-drawing diversity. Spec leaves the bound
  configurable.
- **[Risk] TGCH-TD-S-200-L3-00 ground truth not yet labelled** → the
  regression sub-object reports `expected = 440` but `detected = 0`
  until the ground-truth label file is created. The auditor sees the
  gap explicitly rather than silently.
- **[Risk] HITL ipywidgets UX is brittle for large detection counts**
  → Mitigation: paginated grid (PAGE_SIZE = 20). v2 may move to a
  dedicated app; the schema contract makes the migration cheap.
- **[Trade-off] Per-drawing splits inflate val variance when the
  corpus is small** → Acceptable: the alternative (per-box split)
  leaks structure and gives the model the answers.

## Migration Plan

This is the bootstrap. There is no live system to migrate from; the
prior change's capabilities are superseded textually, not
re-architected at runtime.

Apply-time order (see `tasks.md`):
1. Add real-data ingestion + per-drawing split manifest + hard-neg
   pool manager. Existing synthetic generator remains in place.
2. Add OOD detector + configurable conf/DPI to the inference path.
3. Add metrics emission to `scripts/retrain_yolo.py`. Wire
   regression test against TGCH-TD-S-200-L3-00.
4. Tighten the HITL persistence path (idempotent saves, job
   immutability after corrections exist).
5. Validate the closed loop end-to-end against one real plan.

Rollback: each capability is gated by a constant or config key; if a
new capability regresses behaviour, the constant defaults can revert
to the pre-bootstrap behaviour while a fix is authored.

## Open Questions

- Should the TGCH-TD-S-200-L3-00 ground-truth label file live in this
  repo, or in a separate annotated-corpus repository? (Currently:
  decision deferred to apply time.)
- Should the per-revision metrics emit a Grafana / Prometheus push
  alongside the JSON file, for live dashboarding? (Out of scope for
  v1; revisit if review cadence grows.)
- For the HITL v2 application, do we use a desktop Tauri app or a
  web Streamlit/Gradio app? (Out of scope; schema is the durable
  artefact.)
