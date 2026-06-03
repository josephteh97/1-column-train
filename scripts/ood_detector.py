"""Out-of-distribution hard-failure detector for the column model.

Two cheap signals tell us whether the input is something the model
was actually trained on. If either fires, inference SHOULD abort
rather than emit low-confidence garbage.

  (a) Effective DPI ratio: input DPI / training DPI must sit in
      `[DPI_RATIO_MIN, DPI_RATIO_MAX]` (default `[0.7, 1.4]`).
  (b) Mean per-tile raw detection count must sit in
      `[MEAN_TILE_DETS_MIN, MEAN_TILE_DETS_MAX]` (default `[0.05, 30]`).
      Catches blank pages (below floor) and detection storms from
      drastically miscalibrated input (above ceiling).

Usage:

    from scripts.ood_detector import check_ood, OutOfDistributionError

    try:
        check_ood(input_dpi=300, tile_detection_counts=[12, 14, 9, ...])
    except OutOfDistributionError as e:
        print(f"refusing: {e}")
        sys.exit(2)

The training-time reference DPI is recorded in
`scripts/ood_detector.TRAINING_DPI` (default 300). Adjust per
deployment if the synthetic generator changes scale.
"""
from __future__ import annotations

from typing import Iterable

TRAINING_DPI       = 300
DPI_RATIO_MIN      = 0.7
DPI_RATIO_MAX      = 1.4
MEAN_TILE_DETS_MIN = 0.05
MEAN_TILE_DETS_MAX = 30.0


class OutOfDistributionError(RuntimeError):
    """Raised when an input fails an OOD check."""


def check_dpi(input_dpi: int,
              training_dpi: int = TRAINING_DPI,
              ratio_min: float = DPI_RATIO_MIN,
              ratio_max: float = DPI_RATIO_MAX) -> None:
    if input_dpi <= 0:
        raise OutOfDistributionError(f"input DPI must be positive, got {input_dpi}")
    ratio = input_dpi / training_dpi
    if not (ratio_min <= ratio <= ratio_max):
        lo = int(training_dpi * ratio_min)
        hi = int(training_dpi * ratio_max)
        raise OutOfDistributionError(
            f"input DPI {input_dpi} outside calibrated band "
            f"[{lo}, {hi}] (training DPI {training_dpi}, ratio {ratio:.2f})"
        )


def check_tile_detections(tile_detection_counts: Iterable[int],
                          mean_min: float = MEAN_TILE_DETS_MIN,
                          mean_max: float = MEAN_TILE_DETS_MAX) -> None:
    counts = list(tile_detection_counts)
    if not counts:
        raise OutOfDistributionError("no tiles processed — cannot compute mean detections")
    mean = sum(counts) / len(counts)
    if mean < mean_min:
        raise OutOfDistributionError(
            f"mean tile detections {mean:.3f} below floor {mean_min}: "
            "input looks blank or wildly miscalibrated"
        )
    if mean > mean_max:
        raise OutOfDistributionError(
            f"mean tile detections {mean:.3f} above ceiling {mean_max}: "
            "detector is hallucinating — check rasterisation"
        )


def check_ood(input_dpi: int,
              tile_detection_counts: Iterable[int],
              *,
              training_dpi: int = TRAINING_DPI,
              dpi_ratio_min: float = DPI_RATIO_MIN,
              dpi_ratio_max: float = DPI_RATIO_MAX,
              mean_min: float = MEAN_TILE_DETS_MIN,
              mean_max: float = MEAN_TILE_DETS_MAX) -> None:
    """Run both OOD checks. Raises OutOfDistributionError if either fails."""
    check_dpi(input_dpi, training_dpi, dpi_ratio_min, dpi_ratio_max)
    check_tile_detections(tile_detection_counts, mean_min, mean_max)


if __name__ == "__main__":
    # Smoke test
    check_ood(300, [10, 12, 8, 14])
    print("OK: 300 DPI, mean 11 detections passes")
    try:
        check_ood(150, [10, 12])
    except OutOfDistributionError as e:
        print(f"OK: rejected 150 DPI -> {e}")
    try:
        check_ood(300, [0, 0, 0, 0])
    except OutOfDistributionError as e:
        print(f"OK: rejected blank page -> {e}")
