"""Train the second-stage CNN that re-classifies YOLO bbox crops.

The classifier is the only learned component in the column-review HITL
loop that gets retrained as the reviewer adds corrections. YOLO stays
frozen at the synthetic-baseline `column_detect.pt`. This script builds
its dataset from three sources:

  positives:
    - synthetic column tiles from `dataset/column/{images,labels}/train/`
    - human-drawn FN_ADDED rows in `data/corrections.db`
    - explicit TP confirmations in `data/corrections.db::tp_confirmations`
  negatives:
    - the 24-px-padded FP crops persisted by
      `scripts/hard_negative_pool.py` under `data/hard_negatives/`.

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

def check_prerequisites() -> list[dict]:
    """Return a list of missing-prerequisite dicts, or [] if all present.

    Owned here (not in `column_review/routes/submit.py`) so the path
    constants + fix-command strings stay co-located with the script
    that actually needs them. The UI's `/api/train-classifier` route
    calls this and surfaces the result as a 412 payload.

    Each dict has:
      `code` — machine-readable identifier (UI can dispatch on this)
      `what` — short human description
      `fix`  — copy-paste shell command to resolve
    """
    out: list[dict] = []
    if not SYN_LBL_DIR.is_dir() or not any(SYN_LBL_DIR.glob("*.txt")):
        out.append({
            "code": "synthetic_dataset_missing",
            "what": "synthetic dataset (positive samples)",
            "fix":  "python3 generate_column.py --canvases 30 --no-human-check",
        })
    if not POOL_DIR.is_dir() or not any(POOL_DIR.glob("*.png")):
        out.append({
            "code": "hard_negative_pool_empty",
            "what": "hard-negative pool (FP crops)",
            "fix":  "Mark some false positives in column-review first, "
                    "then run: python3 scripts/hard_negative_pool.py",
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


def _job_render_cached(job_id: str, _cache: dict[str, np.ndarray] = {}
                       ) -> np.ndarray | None:
    """Decode `data/jobs/<job_id>/render.jpg` once per process — the
    cache survives across `_iter_corrections_positives` callers."""
    if job_id in _cache:
        return _cache[job_id]
    rp = JOBS_DIR / job_id / "render.jpg"
    if not rp.is_file():
        return None
    try:
        _cache[job_id] = _load_gray(rp)
        return _cache[job_id]
    except (OSError, ValueError):
        return None


def _job_columns_cached(job_id: str, _cache: dict[str, list] = {}
                        ) -> list | None:
    if job_id in _cache:
        return _cache[job_id]
    pp = JOBS_DIR / job_id / "px_detections.json"
    if not pp.is_file():
        return None
    try:
        _cache[job_id] = json.loads(pp.read_text()).get("columns", [])
        return _cache[job_id]
    except (OSError, json.JSONDecodeError):
        return None


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


def _iter_corrections_positives() -> Iterator[CropSample]:
    """Crop human-drawn FN_ADDED rows + tp_confirmations.

    Both sources use the same lookup: read the bbox from
    `data/jobs/<job_id>/px_detections.json["columns"][element_index]`.
    FN_ADDED appends new rows to that list (so the user-drawn bbox lives
    there); tp_confirmations reference existing detection indices.
    Rescinded deletes are filtered by `iter_effective_corrections` so
    a delete-then-rescind pair does NOT produce a stale positive crop.
    """
    if not CORR_DB.exists():
        return
    conn = sqlite3.connect(str(CORR_DB))
    try:
        # iter_effective_corrections yields:
        #   (job_id, element_type, element_index,
        #    original_element_json, changes_json, is_delete, ts)
        # — the rescind-on-read invariant is enforced at the helper level
        # (groups by (job_id, element_type, element_index) — the full
        # three-part key, not the partial one we'd write inline here).
        fn_rows = [
            (r[0], r[2]) for r in iter_effective_corrections(conn)
            if not r[5]
        ]
        tp_rows = conn.execute(
            "SELECT job_id, element_index FROM tp_confirmations"
        ).fetchall()
    finally:
        conn.close()

    for job_id, idx in fn_rows:
        sample = _crop_at_job_index(job_id, idx, "fn_added")
        if sample is not None:
            yield sample
    for job_id, idx in tp_rows:
        sample = _crop_at_job_index(job_id, idx, "tp_confirm")
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


def _train_epoch(model, opt, samples, *, rng, device):
    import torch
    import torch.nn.functional as F
    model.train()
    n_correct, loss_sum, n = 0, 0.0, 0
    order = list(range(len(samples))); rng.shuffle(order)
    for i in range(0, len(order), _BATCH_SIZE):
        idxs = order[i:i + _BATCH_SIZE]
        x, y = _make_batch(samples, idxs, device, augment_with=rng)
        logits = model(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            preds = (logits >= 0).float()    # sigmoid(0)=0.5 ↔ logit 0
            n_correct += int((preds == y).sum().item())
            loss_sum  += float(loss.item()) * len(idxs)
            n         += len(idxs)
    return loss_sum / max(1, n), n_correct / max(1, n)


def _eval_epoch(model, samples, *, device):
    import torch
    import torch.nn.functional as F
    model.eval()
    n_correct, loss_sum, n = 0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(samples), _BATCH_SIZE):
            idxs = list(range(i, min(i + _BATCH_SIZE, len(samples))))
            x, y = _make_batch(samples, idxs, device, augment_with=None)
            logits = model(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
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

    print("Assembling dataset…", flush=True)
    print("  POSITIVES")
    positives: list[CropSample] = []
    positives.extend(_iter_synthetic_positives())
    n_syn = len(positives)
    positives.extend(_iter_corrections_positives())
    n_corr = len(positives) - n_syn
    print(f"    synthetic: {n_syn}    corrections: {n_corr}")

    print("  NEGATIVES")
    negatives = list(_iter_hard_negative_crops())
    print(f"    hard_negatives: {len(negatives)}")

    if not positives:
        print("\nERROR: no positive samples found.\n"
              "  Run `python3 generate_column.py --canvases 30 "
              "--no-human-check` to populate the synthetic dataset, "
              "or add explicit TP/FN_ADDED rows in column-review.",
              file=sys.stderr)
        sys.exit(2)
    if not negatives:
        print("\nERROR: no negative samples found.\n"
              "  Run `python3 scripts/hard_negative_pool.py` to harvest "
              "FP crops from existing corrections.", file=sys.stderr)
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
    print(f"Training on device={device} for {args.epochs} epoch(s)…",
          flush=True)
    model = _build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state   = None
    bad_epochs   = 0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = _train_epoch(model, opt, train, rng=rng, device=device)
        va_loss, va_acc = _eval_epoch(model, val, device=device)
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
        "n_positives":      sum(s.label for s in samples),
        "n_negatives":      sum(1 for s in samples if s.label == 0),
        "n_synthetic_pos":  n_syn,
        "n_corrections_pos": n_corr,
        "n_hard_neg":       len(negatives),
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
