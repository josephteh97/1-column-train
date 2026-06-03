"""Shared helpers for YOLO dataset writers (generate_column_dataset.py,
retrain_yolo.py). Keeps the directory contract and data.yaml format in
one place so the two scripts can't drift."""

from __future__ import annotations

import shutil
from pathlib import Path


def init_yolo_dataset_dirs(out: Path) -> None:
    """Wipe and recreate `{out}/images/{train,val}` + `{out}/labels/{train,val}`."""
    shutil.rmtree(out, ignore_errors=True)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)


def write_data_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    """Write `{dataset_dir}/data.yaml` in YOLO's expected format."""
    yaml_path = dataset_dir / "data.yaml"
    lines = [
        f"path: {dataset_dir.absolute()}",
        "train: images/train",
        "val: images/val",
        "names:",
        *(f"  {i}: {n}" for i, n in enumerate(class_names)),
    ]
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path
