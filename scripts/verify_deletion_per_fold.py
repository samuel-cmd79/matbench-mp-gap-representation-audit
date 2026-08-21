#!/usr/bin/env python3
"""Print per-fold official-test MAE for the two Level-2 deletions.

Read-only audit: no training and no output files.

Place this file at ``scripts/verify_deletion_per_fold.py`` and run:

    python scripts/verify_deletion_per_fold.py

Deletion delta is ``MAE(group deleted) - MAE(full Level 2)``. Positive values
therefore mean that deletion worsens MAE. Five-fold means are equally weighted;
fold SD uses ``ddof=0``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


FOLDS = range(5)
LABELS_REL = Path("scripts/labels_by_fold.npz")
FULL_REL = Path("matbench_outputs_v2_run0709/predictions_xgb")
DELETION_DIRS = {
    "symmetry": Path("matbench_outputs_v2_ablation_symmetry/predictions_xgb"),
    "coordination": Path(
        "matbench_outputs_v2_ablation_coordination/predictions_xgb"
    ),
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Print deletion MAE by official MatBench fold; write no files."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_root,
        help=f"Repository root (default: {default_root})",
    )
    return parser.parse_args()


def load_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).squeeze()
    if values.ndim != 1:
        raise ValueError(f"{path}: expected a 1-D array, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    if values.min() < -1e-12:
        raise ValueError(f"{path}: expected frozen non-negative predictions")
    return values


def mae(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"{path}: {len(y_pred)} predictions for {len(y_true)} labels"
        )
    return float(np.mean(np.abs(y_true - y_pred), dtype=np.float64))


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    labels_path = root / LABELS_REL
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    rows: dict[str, list[tuple[float, float, float]]] = {
        group: [] for group in DELETION_DIRS
    }

    print("delta_MAE = deleted_MAE - full_Level2_MAE")
    print("Read-only verification: no files are written.\n")
    header = (
        f"{'group':<14} {'fold':>4} {'n_test':>8} "
        f"{'full_MAE':>13} {'deleted_MAE':>13} {'delta_MAE':>13}"
    )
    print(header)
    print("-" * len(header))

    with np.load(labels_path, allow_pickle=False) as labels:
        for fold in FOLDS:
            key = f"y_{fold}"
            if key not in labels.files:
                raise KeyError(f"{labels_path}: missing {key}")
            y_true = np.asarray(labels[key], dtype=np.float64).squeeze()
            if y_true.ndim != 1 or not np.isfinite(y_true).all():
                raise ValueError(f"{labels_path}: invalid {key}")

            full_path = root / FULL_REL / f"pred_fold_{fold}.npy"
            full_mae = mae(y_true, load_vector(full_path), full_path)

            for group, relative_dir in DELETION_DIRS.items():
                deleted_path = root / relative_dir / f"pred_fold_{fold}.npy"
                deleted_mae = mae(
                    y_true, load_vector(deleted_path), deleted_path
                )
                delta = deleted_mae - full_mae
                rows[group].append((full_mae, deleted_mae, delta))
                print(
                    f"{group:<14} {fold:>4d} {len(y_true):>8d} "
                    f"{full_mae:>13.9f} {deleted_mae:>13.9f} "
                    f"{delta:>+13.9f}"
                )

    print("\nEqual-weight five-fold summary (fold SD: ddof=0)")
    summary_header = (
        f"{'group':<14} {'full mean':>13} {'deleted mean':>13} "
        f"{'delta mean':>13} {'delta SD':>13} {'direction':>13}"
    )
    print(summary_header)
    print("-" * len(summary_header))
    for group, group_rows in rows.items():
        values = np.asarray(group_rows, dtype=np.float64)
        deltas = values[:, 2]
        direction = (
            "all positive" if np.all(deltas > 0)
            else "all negative" if np.all(deltas < 0)
            else "mixed"
        )
        print(
            f"{group:<14} {values[:, 0].mean():>13.9f} "
            f"{values[:, 1].mean():>13.9f} {deltas.mean():>+13.9f} "
            f"{deltas.std(ddof=0):>13.9f} {direction:>13}"
        )


if __name__ == "__main__":
    main()
