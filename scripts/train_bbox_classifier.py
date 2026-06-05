"""Train the second-stage CNN that re-classifies YOLO bbox crops.

The classifier is the only learned component in the column-review HITL
loop that gets retrained as the reviewer adds corrections. YOLO stays
frozen at `column_detect.pt` — it cannot regress no matter what we do
to the classifier. The dataset is assembled from all positive sources
on disk; the synthetic dataset is OPTIONAL (used if present, skipped
cleanly if absent — no longer a prerequisite for Train CNN).

  positives:
    - synthetic column tiles from `dataset/column/{images,labels}/train/`
       (OPTIONAL — only contributes if `generate_column.py` has been
       run; absent dataset → skipped silently, training proceeds)
    - human-drawn FN_ADDED rows in `data/corrections.db`
    - explicit TP confirmations in `data/corrections.db::tp_confirmations`
    - IMPLICIT TPs: every model detection in
       `data/jobs/<id>/px_detections.json["columns"]` that the reviewer
       has NOT marked as FP. Safe HERE (unlike YOLO retrain) because
       YOLO stays frozen — worst case the classifier becomes permissive,
       never makes the detector worse. As the reviewer keeps marking
       FPs across drawings, the implicit-TP set sharpens.
  negatives:
    - the 24-px-padded FP crops persisted by
       `scripts/hard_negative_pool.py` under `data/hard_negatives/`.

Class imbalance is handled by `pos_weight = n_neg / n_pos` in the BCE
loss so the classifier doesn't collapse to "accept everything" when
implicit TPs vastly outnumber the curated FPs.

Runs end-to-end on CPU in under a minute. Output:
  column_classifier.pt          — state_dict for `BBoxClassifier`
  column_classifier.meta.json   — counts + val accuracy + training duration

Usage:
  python3 scripts/train_bbox_classifier.py                 # defaults
  python3 scripts/train_bbox_classifier.py --epochs 50
  python3 scripts/train_bbox_classifier.py --dry-run       # assemble only
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

# Trust local files; A0 renders trip PIL's default ~89 MP guard.
Image.MAX_IMAGE_PIXELS = None

# Anchor everything to the project root. JOBS_DIR + DB_PATH come from
# `scripts/corrections_logger.py` (the canonical owners) and POOL_DIR
# from `scripts/hard_negative_pool.py` so the data-tree layout is owned
# in one place per concern — moving the data root only requires editing
# those two files.
_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.corrections_logger import (    # noqa: E402
    DB_PATH as CORR_DB,
    JOBS_DIR,
    iter_effective_corrections,
)
from scripts.hard_negative_pool import POOL_DIR   # noqa: E402

SYN_IMG_DIR = _PROJECT_ROOT / "dataset" / "column" / "images" / "train"
SYN_LBL_DIR = _PROJECT_ROOT / "dataset" / "column" / "labels" / "train"

# Re-use the package's crop geometry so train and inference agree.
from column_review.bbox_classifier import (   # noqa: E402
    CROP_MARGIN_PX, CLASSIFIER_SIZE,
    _build_model, crop_64x64,
)


@dataclass
class CropSample:
    """One 64×64 uint8 crop + binary label. Source field is for the
    training log only (audit which sources actually contributed)."""
    crop: np.ndarray
    label: int
    source: str   # "synthetic" / "fn_added" / "tp_confirm" / "hard_neg"


# ────────────────────────────────────────────────────────────────────────
# Cropping primitive `crop_64x64` is imported from
# `column_review.bbox_classifier` above so train and inference share one
# implementation; drift would mean different distributions silently.
# ────────────────────────────────────────────────────────────────────────

def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


# ────────────────────────────────────────────────────────────────────────
# Prerequisite check (shared with the column-review web UI)
# ────────────────────────────────────────────────────────────────────────

def _iter_candidate_job_dirs() -> Iterator[Path]:
    """Yield every `data/jobs/<id>/` that has BOTH render.jpg AND
    px_detections.json on disk. Single definition of "implicit-TP-eligible
    job" — the probe and the iterator share it so a tightening here
    can't leave one of them stale."""
    if not JOBS_DIR.is_dir():
        return
    for job_dir in sorted(JOBS_DIR.iterdir()):
        if not job_dir.is_dir():
            continue
        if not (job_dir / "render.jpg").is_file():
            continue
        if not (job_dir / "px_detections.json").is_file():
            continue
        yield job_dir


def _has_any_implicit_tp_source(fp_set: set[tuple[str, int]],
                                tp_set: set[tuple[str, int]]) -> bool:
    """True iff some candidate job has at least one row that survives the
    same filters `_iter_implicit_tp_positives` applies (not human_added,
    not FP-marked, not already an explicit TP-confirm, bbox shape OK).

    A True here guarantees the iterator yields at least one sample —
    without the iterator-matching filter, the probe would pass for
    jobs whose only detections are all-FP-marked or all-human_added,
    and `_iter_implicit_tp_positives` would then yield zero, defeating
    the prereq's purpose.
    """
    for job_dir in _iter_candidate_job_dirs():
        job_id = job_dir.name
        cols = _job_columns_cached(job_id)
        if not cols:
            continue
        for idx, row in enumerate(cols):
            if not isinstance(row, dict):
                continue
            if row.get("source") == "human_added":
                continue
            if (job_id, idx) in fp_set or (job_id, idx) in tp_set:
                continue
            bbox = row.get("bbox")
            if bbox and len(bbox) >= 4:
                return True
    return False


def check_prerequisites() -> list[dict]:
    """Return a list of missing-prerequisite dicts, or [] if all present.

    Owned here (not in `column_review/routes/train.py`) so the path
    constants + fix-command strings stay co-located with the script
    that actually needs them. The UI's `/api/train-classifier` route
    calls this and surfaces the result as a 412 payload.

    Each dict has:
      `code` — machine-readable identifier (UI can dispatch on this)
      `what` — short human description
      `fix`  — copy-paste shell command to resolve

    The synthetic dataset is OPTIONAL — its absence is no longer a
    prereq failure. As long as ONE positive source is available
    (synthetic OR FN_ADDED OR TP-confirm OR an actually-survivable
    implicit TP from a job on disk) AND there's a negative class
    (FP crops), training proceeds.
    """
    out: list[dict] = []
    if not POOL_DIR.is_dir() or not any(POOL_DIR.glob("*.png")):
        out.append({
            "code": "hard_negative_pool_empty",
            "what": "hard-negative pool (FP crops)",
            "fix":  "Mark some false positives in column-review first, "
                    "then run: python3 scripts/hard_negative_pool.py",
        })
    has_synthetic = SYN_LBL_DIR.is_dir() and any(SYN_LBL_DIR.glob("*.txt"))
    # Load correction state so the probe sees the SAME filters the
    # iterators apply at training time (fp_set, tp_set). Without this,
    # the probe greenlights jobs whose only detections are all-FP-marked
    # or all-human_added, then training assembles zero positives and the
    # runtime guard at the top of main()'s assembly section exits 2.
    fp_set, fn_rows, tp_rows = _load_correction_state()
    tp_set = {(j, i) for j, i in tp_rows}
    has_explicit_pos = bool(fn_rows) or bool(tp_rows)
    has_implicit = _has_any_implicit_tp_source(fp_set, tp_set)
    if not (has_synthetic or has_explicit_pos or has_implicit):
        out.append({
            "code": "no_positive_source",
            "what": "no positive source (no synthetic dataset, no human-drawn FN_ADDED,"
                    " no TP confirmations, no inferred-and-unmarked detections)",
            "fix":  "In column-review, open a drawing, click Run YOLO, then "
                    "either leave some detections un-clicked (they become "
                    "implicit positives) or draw a missed column with FN-mode "
                    "drag (explicit FN_ADDED).",
        })
    return out


# ────────────────────────────────────────────────────────────────────────
# Positive sources
# ────────────────────────────────────────────────────────────────────────

def _iter_synthetic_positives() -> Iterator[CropSample]:
    """Crop every YOLO label box from the synthetic training tiles."""
    if not SYN_IMG_DIR.is_dir() or not SYN_LBL_DIR.is_dir():
        print(f"  no synthetic dataset at {SYN_IMG_DIR.parent} — "
              "run `python3 generate_column.py --canvases 30 "
              "--no-human-check` to populate it", flush=True)
        return
    label_files = sorted(SYN_LBL_DIR.glob("*.txt"))
    if not label_files:
        print(f"  no synthetic labels under {SYN_LBL_DIR}", flush=True)
        return
    print(f"  reading {len(label_files)} synthetic tile(s)…", flush=True)
    for lbl_path in label_files:
        img_path = SYN_IMG_DIR / (lbl_path.stem + ".png")
        if not img_path.is_file():
            continue
        try:
            img_gray = _load_gray(img_path)
        except (OSError, ValueError):
            continue
        H, W = img_gray.shape
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cxn, cyn, wn, hn = parts
            try:
                cx, cy, w, h = float(cxn) * W, float(cyn) * H, float(wn) * W, float(hn) * H
            except ValueError:
                continue
            bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            yield CropSample(crop_64x64(img_gray, bbox), 1, "synthetic")


# Module-level mtime-keyed caches. The previous `_cache: dict = {}`
# default-arg pattern aliased a SINGLE dict across every call across
# every Train CNN click in the FastAPI parent process, so a /api/infer
# that overwrote px_detections.json on disk was invisible to a later
# prereq probe. Storing (mtime, value) lets us cheaply detect rewrites.
_RENDER_CACHE: dict[str, tuple[float, np.ndarray]] = {}
_COLUMNS_CACHE: dict[str, tuple[float, list]] = {}


def _job_render_cached(job_id: str) -> np.ndarray | None:
    """Decode `data/jobs/<job_id>/render.jpg` and cache by mtime — the
    /api/infer route rewrites this file in place, so the cache MUST
    invalidate on mtime change to avoid serving stale pixels."""
    rp = JOBS_DIR / job_id / "render.jpg"
    if not rp.is_file():
        return None
    try:
        mtime = rp.stat().st_mtime
    except OSError:
        return None
    cached = _RENDER_CACHE.get(job_id)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        arr = _load_gray(rp)
    except (OSError, ValueError):
        return None
    _RENDER_CACHE[job_id] = (mtime, arr)
    return arr


def _job_columns_cached(job_id: str) -> list | None:
    """Read columns[] from `data/jobs/<job_id>/px_detections.json` and
    cache by mtime — see `_job_render_cached` for the cache-staleness
    rationale (px_detections.json is rewritten on every /api/infer)."""
    pp = JOBS_DIR / job_id / "px_detections.json"
    if not pp.is_file():
        return None
    try:
        mtime = pp.stat().st_mtime
    except OSError:
        return None
    cached = _COLUMNS_CACHE.get(job_id)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        cols = json.loads(pp.read_text()).get("columns", [])
    except (OSError, json.JSONDecodeError):
        return None
    _COLUMNS_CACHE[job_id] = (mtime, cols)
    return cols


def _crop_at_job_index(job_id: str, idx: int, source: str
                       ) -> CropSample | None:
    """Look up `px_detections.json["columns"][idx]["bbox"]` for the job
    and crop it from the render. Returns None on any missing piece
    (job not on disk, idx out of range, malformed row)."""
    cols = _job_columns_cached(job_id)
    if cols is None or idx < 0 or idx >= len(cols):
        return None
    row = cols[idx]
    bbox = row.get("bbox") if isinstance(row, dict) else None
    if not bbox or len(bbox) < 4:
        return None
    img = _job_render_cached(job_id)
    if img is None:
        return None
    return CropSample(crop_64x64(img, bbox), 1, source)


def _load_correction_state() -> tuple[set[tuple[str, int]],
                                       list[tuple[str, int]],
                                       list[tuple[str, int]]]:
    """Single pass over the corrections DB returning everything the
    three positive iterators need: (fp_set, fn_rows, tp_rows).

    - `fp_set`: {(job_id, element_index), …} for effective is_delete=1
      rows. Consumed by `_iter_implicit_tp_positives` to skip FP-marked
      detections.
    - `fn_rows`: [(job_id, element_index), …] for effective is_delete=0
      rows. Consumed by `_iter_fn_added_positives`.
    - `tp_rows`: [(job_id, element_index), …] from the `tp_confirmations`
      sidecar table. Consumed by `_iter_tp_confirm_positives`.

    Rescind-on-read is enforced by `iter_effective_corrections` so a
    delete-then-rescind pair does NOT appear in either fp_set or fn_rows.
    """
    fp_set: set[tuple[str, int]] = set()
    fn_rows: list[tuple[str, int]] = []
    tp_rows: list[tuple[str, int]] = []
    if not CORR_DB.exists():
        return fp_set, fn_rows, tp_rows
    conn = sqlite3.connect(str(CORR_DB))
    try:
        # iter_effective_corrections yields:
        #   (job_id, element_type, element_index,
        #    original_element_json, changes_json, is_delete, ts)
        for r in iter_effective_corrections(conn):
            key = (r[0], int(r[2]))
            (fp_set.add(key) if r[5] else fn_rows.append(key))
        tp_rows = conn.execute(
            "SELECT job_id, element_index FROM tp_confirmations"
        ).fetchall()
    finally:
        conn.close()
    return fp_set, fn_rows, tp_rows


def _iter_corrections_at_indices(rows: list[tuple[str, int]],
                                 source: str) -> Iterator[CropSample]:
    """Crop each `(job_id, element_index)` via `_crop_at_job_index`.
    Used by both the FN_ADDED and TP-confirm paths so they share one
    lookup + crop pipeline."""
    for job_id, idx in rows:
        sample = _crop_at_job_index(job_id, idx, source)
        if sample is not None:
            yield sample


def _iter_implicit_tp_positives(fp_set: set[tuple[str, int]],
                                tp_set: set[tuple[str, int]] | None = None,
                                ) -> Iterator[CropSample]:
    """Every model-source detection NOT marked as FP and NOT
    `source='human_added'` is an implicit positive.

    Safe for the classifier (unlike YOLO retrain) because YOLO stays
    frozen — wrong implicit positives only push the classifier toward
    "accept everything", never make the detector regress. The user
    accepts this trade-off and re-trains as more FPs are clicked.

    Skips human_added rows (those are already covered by the FN_ADDED
    iterator). Skips FP-marked rows via `fp_set`. Skips rows that are
    explicit TP-confirms via `tp_set` — those are already yielded by
    `_iter_corrections_at_indices(tp_rows, 'tp_confirm')`; without
    this guard the same crop would appear twice in `positives` and
    leak across the train/val split.

    The render decode happens only AFTER confirming at least one row
    survives the filters — the render is the dominant cost.
    """
    tp_set = tp_set or set()
    for job_dir in _iter_candidate_job_dirs():
        job_id = job_dir.name
        cols = _job_columns_cached(job_id)
        if not cols:
            continue
        survivors = [
            idx for idx, row in enumerate(cols)
            if isinstance(row, dict)
            and row.get("source") != "human_added"
            and (job_id, idx) not in fp_set
            and (job_id, idx) not in tp_set
        ]
        if not survivors:
            continue
        for idx in survivors:
            sample = _crop_at_job_index(job_id, idx, "implicit_tp")
            if sample is not None:
                yield sample


# ────────────────────────────────────────────────────────────────────────
# Negative source (already cropped on disk)
# ────────────────────────────────────────────────────────────────────────

def _iter_hard_negative_crops() -> Iterator[CropSample]:
    if not POOL_DIR.is_dir():
        return
    pngs = sorted(POOL_DIR.glob("*.png"))
    print(f"  reading {len(pngs)} hard-negative crop(s)…", flush=True)
    import cv2
    for p in pngs:
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("L"))
        except (OSError, ValueError):
            continue
        if arr.size == 0:
            continue
        resized = cv2.resize(
            arr, (CLASSIFIER_SIZE, CLASSIFIER_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        yield CropSample(resized, 0, "hard_neg")


# ────────────────────────────────────────────────────────────────────────
# Augmentation (rotations + hflip + brightness)
# ────────────────────────────────────────────────────────────────────────

def _augment(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    """Columns are square + axis-aligned → 4 rotations × 2 flips ≡ free
    augmentation. Brightness ±10% mimics raster/JPEG drift.

    Contiguity isn't enforced here — the per-batch `np.stack` in
    `_iter_batches` always returns a contiguous output, so anything
    fed to `torch.from_numpy` downstream is already C-ordered.
    """
    k = rng.randint(0, 3)
    if k:
        crop = np.rot90(crop, k=k)
    if rng.random() < 0.5:
        crop = np.fliplr(crop)
    delta = rng.uniform(-25.5, 25.5)
    return np.clip(crop.astype(np.int16) + int(delta), 0, 255).astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────
# Train / eval
# ────────────────────────────────────────────────────────────────────────

def _split(samples: list[CropSample], val_frac: float, seed: int):
    pos = [s for s in samples if s.label == 1]
    neg = [s for s in samples if s.label == 0]
    rng = random.Random(seed)
    rng.shuffle(pos); rng.shuffle(neg)
    cut_p = max(1, int(len(pos) * val_frac))
    cut_n = max(1, int(len(neg) * val_frac))
    val   = pos[:cut_p] + neg[:cut_n]
    train = pos[cut_p:] + neg[cut_n:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val


_BATCH_SIZE = 64


def _make_batch(samples, idxs, device, *, augment_with: random.Random | None):
    """Stack a batch of samples → (x, y) on `device`.

    uint8 stays on CPU through `np.stack`/`torch.from_numpy`, hits the
    device as uint8, then casts to float32 + scales on the device. Drops
    the CPU float32 peak by 4× vs. an upfront `.float().div_(255.0)`.
    """
    import torch
    crops_np = np.stack([
        _augment(samples[j].crop, augment_with) if augment_with else samples[j].crop
        for j in idxs
    ], axis=0)
    labels_np = np.array([samples[j].label for j in idxs], dtype=np.float32)
    x = torch.from_numpy(crops_np).to(device, non_blocking=True)
    x = x.unsqueeze(1).float().div_(255.0)
    y = torch.from_numpy(labels_np).to(device, non_blocking=True)
    return x, y


def _train_epoch(model, opt, samples, *, rng, device, pos_weight):
    """pos_weight: torch.Tensor[1] on `device`. Scales the positive class
    in BCE so imbalanced datasets (lots of implicit TPs vs. 121 FPs)
    don't collapse the classifier to "accept everything"."""
    import torch
    import torch.nn.functional as F
    model.train()
    n_correct, loss_sum, n = 0, 0.0, 0
    order = list(range(len(samples))); rng.shuffle(order)
    for i in range(0, len(order), _BATCH_SIZE):
        idxs = order[i:i + _BATCH_SIZE]
        x, y = _make_batch(samples, idxs, device, augment_with=rng)
        logits = model(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            preds = (logits >= 0).float()    # sigmoid(0)=0.5 ↔ logit 0
            n_correct += int((preds == y).sum().item())
            loss_sum  += float(loss.item()) * len(idxs)
            n         += len(idxs)
    return loss_sum / max(1, n), n_correct / max(1, n)


def _eval_epoch(model, samples, *, device, pos_weight):
    import torch
    import torch.nn.functional as F
    model.eval()
    n_correct, loss_sum, n = 0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(samples), _BATCH_SIZE):
            idxs = list(range(i, min(i + _BATCH_SIZE, len(samples))))
            x, y = _make_batch(samples, idxs, device, augment_with=None)
            logits = model(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            preds = (logits >= 0).float()
            n_correct += int((preds == y).sum().item())
            loss_sum  += float(loss.item()) * len(idxs)
            n         += len(idxs)
    return loss_sum / max(1, n), n_correct / max(1, n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",   type=int, default=30)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stop after N epochs of no val-acc gain.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Assemble dataset only; skip training.")
    parser.add_argument("--device",   default=None,
                        help="'cpu' or 'cuda:0' (auto-detected by default).")
    parser.add_argument("--output",
                        default=str(_PROJECT_ROOT / "column_classifier.pt"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    t0 = time.perf_counter()

    # Single source of truth for prereq messaging — the web UI's
    # `/api/train-classifier` 412 payload comes from this same function,
    # so CLI failures speak with the same voice.
    missing = check_prerequisites()
    if missing:
        print("\nERROR: cannot start training — prerequisites missing:",
              file=sys.stderr)
        for m in missing:
            print(f"  • {m['what']}\n      fix: {m['fix']}",
                  file=sys.stderr)
        sys.exit(2)

    # One pass over corrections.db serves all three downstream iterators:
    # explicit FN_ADDED, explicit TP-confirm, and the fp_set filter used
    # by the implicit-TP iterator. tp_set additionally lets the implicit
    # iterator skip rows already yielded as explicit TP-confirms (no
    # double-counting).
    fp_set, fn_rows, tp_rows = _load_correction_state()
    tp_set = {(j, i) for j, i in tp_rows}

    print("Assembling dataset…", flush=True)
    print("  POSITIVES")
    syn_pos      = list(_iter_synthetic_positives())
    fn_pos       = list(_iter_corrections_at_indices(fn_rows, "fn_added"))
    tp_pos       = list(_iter_corrections_at_indices(tp_rows, "tp_confirm"))
    implicit_pos = list(_iter_implicit_tp_positives(fp_set, tp_set))
    positives    = syn_pos + fn_pos + tp_pos + implicit_pos
    print(f"    synthetic: {len(syn_pos)}    fn_added: {len(fn_pos)}    "
          f"tp_confirm: {len(tp_pos)}    implicit_tp: {len(implicit_pos)}")

    print("  NEGATIVES")
    negatives = list(_iter_hard_negative_crops())
    print(f"    hard_negatives: {len(negatives)}")

    # Defense in depth — check_prerequisites() above is the FAST
    # pre-flight (no crop work). The actual sample assembly can still
    # collapse to zero if every render.jpg fails to decode, every bbox
    # is malformed, or every hard-negative PNG fails to load. In that
    # case the prereq probe greenlit a state that won't actually train,
    # and we must NOT overwrite column_classifier.pt with a degenerate
    # model.
    if not positives:
        print("\nERROR: no positive samples materialised after assembly.\n"
              "  Prereq passed but every positive source produced zero "
              "crops (likely: render.jpg failed to decode or every bbox "
              "is malformed).\n"
              "  Inspect data/jobs/<id>/render.jpg and px_detections.json.",
              file=sys.stderr)
        sys.exit(2)
    if not negatives:
        print("\nERROR: no negative samples materialised after assembly.\n"
              "  data/hard_negatives/*.png exist but every file failed to "
              "decode. Inspect the PNGs and re-run "
              "`python3 scripts/hard_negative_pool.py` if the manifest is "
              "stale.", file=sys.stderr)
        sys.exit(2)

    samples = positives + negatives
    train, val = _split(samples, args.val_frac, args.seed)
    n_pos = sum(s.label for s in train)
    n_neg = len(train) - n_pos
    print(f"  train: {len(train)} ({n_pos} pos / {n_neg} neg)   "
          f"val: {len(val)}")

    if args.dry_run:
        print("(dry-run — skipping training.)")
        return

    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        sys.exit(1)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    # Class-imbalance compensation. With implicit TPs in the mix the
    # positive class typically outnumbers negatives 5-10×; pos_weight
    # pushes the BCE loss to weigh the (rarer) negative class enough
    # that the classifier doesn't collapse to "accept everything".
    pos_weight_value = len(negatives) / max(1, len(positives))
    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(f"Training on device={device} for {args.epochs} epoch(s) "
          f"(pos_weight={pos_weight_value:.3f})…", flush=True)
    model = _build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state   = None
    bad_epochs   = 0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = _train_epoch(
            model, opt, train, rng=rng, device=device, pos_weight=pos_weight)
        va_loss, va_acc = _eval_epoch(
            model, val, device=device, pos_weight=pos_weight)
        print(f"  epoch {ep:>2}/{args.epochs}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f}   "
              f"val loss={va_loss:.4f} acc={va_acc:.3f}",
              flush=True)
        if va_acc > best_val_acc + 1e-4:
            best_val_acc = va_acc
            best_state   = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            bad_epochs   = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early-stop at epoch {ep} "
                      f"(no val-acc gain for {args.patience} epochs)")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}

    out_path = Path(args.output)
    torch.save(best_state, str(out_path))
    elapsed = time.perf_counter() - t0
    meta = {
        "n_train":          len(train),
        "n_val":            len(val),
        "n_positives":      len(positives),
        "n_negatives":      len(negatives),
        "n_synthetic_pos":  len(syn_pos),
        "n_fn_added":       len(fn_pos),
        "n_tp_confirm":     len(tp_pos),
        "n_implicit_tp":    len(implicit_pos),
        "n_hard_neg":       len(negatives),
        "pos_weight":       round(pos_weight_value, 4),
        "best_val_acc":     float(best_val_acc),
        "epochs_trained":   ep,
        "duration_seconds": round(elapsed, 1),
        "device":           device,
        "crop_size":        CLASSIFIER_SIZE,
        "crop_margin_px":   CROP_MARGIN_PX,
        "saved_ts":         time.time(),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved: {out_path}  ({elapsed:.1f}s, val_acc={best_val_acc:.3f})")
    print(f"Meta : {out_path.with_suffix('.meta.json')}")
    print("\nNext: restart column-review — the classifier will be picked "
          "up automatically on the next /api/infer call.")


if __name__ == "__main__":
    main()
