#!/usr/bin/env python3
"""
Compare frozen MatBench mp-gap XGBoost predictions for:

    1. Level 1 (132 Magpie composition features)
    2. Level 1 + BandCenter (133 features)
    3. Level 2 (283 features)
    4. Level 3 (feature count not applicable to the saved prediction artifact)

This script performs no training and no feature generation. It reads the
already-clipped frozen predictions, binds the legacy .npy arrays to the official
MatBench test-fold sample order, validates the ID/fold/label alignment, and
writes the aggregate, per-fold, paired-difference, subset, and audit outputs to
a new directory.

Defaults reflect the supplied Mac project layout:

    Level 1:              ../outputs_v1_run0709
    Level 1 + BandCenter: ../matbench_outputs_l1_plus_bandcenter_run0731
    Level 2:              ../matbench_outputs_v2_run0709
    Level 3:              ../results_v4

For Level 2, the historical extra-underscore directory
``../matbench_outputs_v2_run_0709`` is automatically tried when the release
directory is absent.

Example
-------
    python compare_l1_bandcenter_l2_l3.py

Explicit paths:
    python compare_l1_bandcenter_l2_l3.py \
        --l1-results-dir ../outputs_v1_run0709 \
        --l1bc-results-dir ../matbench_outputs_l1_plus_bandcenter_run0731 \
        --l2-results-dir ../matbench_outputs_v2_run0709 \
        --l3-results-dir ../results_v4 \
        --output-dir ../matbench_outputs_l1_bandcenter_l2_l3_comparison_run0731_ddof0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_NAME = "matbench_mp_gap"
EXPECTED_FOLDS = (0, 1, 2, 3, 4)

MODEL_SPECS = (
    {
        "key": "l1",
        "label": "Level 1",
        "features": 132,
        "prediction_column": "prediction_level1",
    },
    {
        "key": "l1bc",
        "label": "Level 1 + BandCenter",
        "features": 133,
        "prediction_column": "prediction_level1_plus_bandcenter",
    },
    {
        "key": "l2",
        "label": "Level 2",
        "features": 283,
        "prediction_column": "prediction_level2",
    },
    {
        "key": "l3",
        "label": "Level 3",
        "features": "N/A",
        "prediction_column": "prediction_level3",
    },
)

DEFAULT_L1_RESULTS_DIR = Path("../outputs_v1_run0709")
DEFAULT_L1BC_RESULTS_DIR = Path("../matbench_outputs_l1_plus_bandcenter_run0731")
DEFAULT_L2_RESULTS_DIR = Path("../matbench_outputs_v2_run0709")
DEFAULT_L2_FALLBACK_DIR = Path("../matbench_outputs_v2_run_0709")
DEFAULT_L3_RESULTS_DIR = Path("../results_v4")
DEFAULT_OUTPUT_DIR = Path(
    "../matbench_outputs_l1_bandcenter_l2_l3_comparison_run0731_ddof0"
)

L1BC_PREDICTIONS_NAME = "l1_plus_bandcenter_predictions.csv"
L1BC_CONFIG_NAME = "l1_plus_bandcenter_config.json"

SUMMARY_NAME = "l1_bandcenter_l2_l3_comparison.csv"
PER_FOLD_NAME = "l1_bandcenter_l2_l3_per_fold.csv"
ALIGNED_PREDICTIONS_NAME = "l1_bandcenter_l2_l3_aligned_predictions.csv"
PAIRED_DIFFERENCES_NAME = "l1_bandcenter_l2_l3_paired_mae_differences.csv"
PAIRED_SUMMARY_NAME = "l1_bandcenter_l2_l3_paired_difference_summary.csv"
SEQUENTIAL_GAINS_NAME = "l1_bandcenter_l2_l3_sequential_gains.csv"
SUBSET_METRICS_NAME = "l1_bandcenter_l2_l3_subset_metrics.csv"
FOLD_ASSIGNMENTS_NAME = "l1_bandcenter_l2_l3_fold_assignments.csv"
SOURCE_MANIFEST_NAME = "l1_bandcenter_l2_l3_source_manifest.csv"
CONFIG_NAME = "l1_bandcenter_l2_l3_comparison_config.json"
AUDIT_NAME = "l1_bandcenter_l2_l3_comparison_audit.md"
LOG_NAME = "l1_bandcenter_l2_l3_comparison_run.log"


# Loaded only after argparse has processed --help.
np = None
pd = None
mean_absolute_error = None
mean_squared_error = None
r2_score = None
MatbenchBenchmark = None


class TeeStream:
    """Write a stream to both terminal and log."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def fileno(self):
        return self.terminal.fileno()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen Level-1, Level-1+BandCenter, Level-2, and Level-3 "
            "MatBench mp-gap XGBoost predictions."
        )
    )
    parser.add_argument(
        "--l1-results-dir",
        type=Path,
        default=DEFAULT_L1_RESULTS_DIR,
        help=f"Frozen Level-1 result directory (default: {DEFAULT_L1_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--l1bc-results-dir",
        type=Path,
        default=DEFAULT_L1BC_RESULTS_DIR,
        help=(
            "Completed Level-1+BandCenter result directory "
            f"(default: {DEFAULT_L1BC_RESULTS_DIR})."
        ),
    )
    parser.add_argument(
        "--l2-results-dir",
        type=Path,
        default=DEFAULT_L2_RESULTS_DIR,
        help=(
            "Frozen Level-2 result directory "
            f"(default: {DEFAULT_L2_RESULTS_DIR}; historical fallback: "
            f"{DEFAULT_L2_FALLBACK_DIR})."
        ),
    )
    parser.add_argument(
        "--l3-results-dir",
        type=Path,
        default=DEFAULT_L3_RESULTS_DIR,
        help=(
            "Frozen Level-3 result directory containing "
            "fold_{0..4}/test_preds_clipped.npz "
            f"(default: {DEFAULT_L3_RESULTS_DIR})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"New comparison output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    global np, pd
    global mean_absolute_error, mean_squared_error, r2_score
    global MatbenchBenchmark

    try:
        import numpy as _np
        import pandas as _pd
        from matbench.bench import MatbenchBenchmark as _MatbenchBenchmark
        from sklearn.metrics import (
            mean_absolute_error as _mean_absolute_error,
            mean_squared_error as _mean_squared_error,
            r2_score as _r2_score,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Activate the same Python environment "
            "used for the MatBench runs before executing this script. "
            f"Original import error: {exc}"
        ) from exc

    np = _np
    pd = _pd
    mean_absolute_error = _mean_absolute_error
    mean_squared_error = _mean_squared_error
    r2_score = _r2_score
    MatbenchBenchmark = _MatbenchBenchmark


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not-installed-or-not-detected"


def collect_versions() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "matbench": package_version("matbench"),
        "matminer": package_version("matminer"),
        "pymatgen": package_version("pymatgen"),
        "scikit-learn": package_version("scikit-learn", "sklearn"),
        "xgboost": package_version("xgboost"),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if np is not None and isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        if not (value == value) or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(data: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def atomic_write_csv(frame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"Output path exists but is not a directory: {path}")
        if any(path.iterdir()):
            raise RuntimeError(
                f"Comparison output directory is not empty: {path}\n"
                "Choose a new --output-dir. Existing model outputs are never modified."
            )
    path.mkdir(parents=True, exist_ok=True)


def resolve_input_dir(
    requested: Path,
    label: str,
    fallback: Path | None = None,
) -> Path:
    requested = requested.expanduser()
    if requested.is_dir():
        return requested
    if fallback is not None and fallback.expanduser().is_dir():
        resolved = fallback.expanduser()
        print(
            f"{label}: requested directory {requested} was not found; "
            f"using compatible historical path {resolved}."
        )
        return resolved
    candidates = [str(requested)]
    if fallback is not None:
        candidates.append(str(fallback.expanduser()))
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"{label} result directory was not found. Checked:\n{formatted}"
    )


def get_task() -> Any:
    benchmark = MatbenchBenchmark(autoload=False)
    task = next(
        (candidate for candidate in benchmark.tasks if candidate.dataset_name == TASK_NAME),
        None,
    )
    if task is None:
        raise RuntimeError(f"MatBench task not found: {TASK_NAME}")
    task.load()
    folds = tuple(int(value) for value in task.folds)
    if folds != EXPECTED_FOLDS:
        raise AssertionError(
            f"Expected official folds {EXPECTED_FOLDS}, found {folds}."
        )
    return task


def series_ids(series: Any, label: str) -> list[str]:
    if not hasattr(series, "index"):
        raise AssertionError(f"{label} has no pandas index.")
    ids = [str(value) for value in series.index.tolist()]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{label} contains duplicate sample IDs.")
    return ids


def labels_to_numpy(labels: Any, ids: list[str], label: str):
    if hasattr(labels, "index"):
        label_ids = [str(value) for value in labels.index.tolist()]
        if label_ids != ids:
            raise AssertionError(f"{label} label index does not match input IDs.")
    values = np.asarray(labels, dtype=float)
    if values.ndim != 1 or len(values) != len(ids):
        raise AssertionError(
            f"{label} labels have shape {values.shape}; expected ({len(ids)},)."
        )
    if not np.isfinite(values).all():
        raise AssertionError(f"{label} labels contain NaN/infinity.")
    return values


def load_official_fold_metadata(
    task: Any,
) -> tuple[dict[int, dict[str, Any]], Any, str]:
    metadata = {}
    reference_universe = None
    all_test_ids = []
    assignment_rows = []

    for fold in EXPECTED_FOLDS:
        train_inputs, _ = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
        train_ids = series_ids(train_inputs, f"fold {fold} official train")
        test_ids = series_ids(test_inputs, f"fold {fold} official test")
        test_y = labels_to_numpy(test_outputs, test_ids, f"fold {fold} official test")

        train_set = set(train_ids)
        test_set = set(test_ids)
        if train_set & test_set:
            raise AssertionError(f"Fold {fold} official train/test IDs overlap.")
        universe = train_set | test_set
        if reference_universe is None:
            reference_universe = universe
        elif universe != reference_universe:
            raise AssertionError(
                f"Fold {fold} sample universe differs from fold 0."
            )

        all_test_ids.extend(test_ids)
        for position, (sample_id, y_true) in enumerate(zip(test_ids, test_y)):
            assignment_rows.append(
                {
                    "sample_id": sample_id,
                    "official_fold": fold,
                    "official_test_position": position,
                    "true_label": float(y_true),
                }
            )
        metadata[fold] = {
            "test_ids": test_ids,
            "test_y": test_y,
        }

    if len(all_test_ids) != len(set(all_test_ids)):
        raise AssertionError("A sample appears in multiple official test folds.")
    if set(all_test_ids) != reference_universe:
        raise AssertionError(
            "The five official test folds do not cover the full sample universe exactly once."
        )

    assignments = pd.DataFrame(assignment_rows).sort_values(
        ["official_fold", "official_test_position"], kind="stable"
    )
    hash_frame = assignments[["sample_id", "official_fold"]].sort_values(
        ["sample_id", "official_fold"], kind="stable"
    )
    canonical = "".join(
        f"{row.sample_id},{int(row.official_fold)}\n"
        for row in hash_frame.itertuples(index=False)
    )
    assignment_hash = sha256_text(canonical)
    return metadata, assignments, assignment_hash


def assert_clipped_predictions(values, label: str) -> None:
    if values.ndim != 1:
        raise AssertionError(f"{label} predictions are not one-dimensional.")
    if not np.isfinite(values).all():
        raise AssertionError(f"{label} predictions contain NaN/infinity.")
    if (values < 0).any():
        minimum = float(np.min(values))
        raise AssertionError(
            f"{label} predictions contain negative values (minimum={minimum}); "
            "expected frozen non-negativity-clipped predictions."
        )


def load_legacy_npy_predictions(
    results_dir: Path,
    model_label: str,
    fold_metadata: dict[int, dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    prediction_dir = results_dir / "predictions_xgb"
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"{model_label} XGBoost prediction directory is missing: {prediction_dir}"
        )

    frames = []
    manifest = []
    for fold in EXPECTED_FOLDS:
        path = prediction_dir / f"pred_fold_{fold}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"{model_label} frozen prediction file is missing: {path}"
            )
        values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        if values.ndim != 1:
            raise AssertionError(
                f"{model_label} fold {fold} prediction array has shape "
                f"{values.shape}; expected a one-dimensional array."
            )
        ids = fold_metadata[fold]["test_ids"]
        y_true = fold_metadata[fold]["test_y"]
        if len(values) != len(ids):
            raise AssertionError(
                f"{model_label} fold {fold} has {len(values)} predictions but "
                f"the official test fold has {len(ids)} samples."
            )
        assert_clipped_predictions(values, f"{model_label} fold {fold}")

        frames.append(
            pd.DataFrame(
                {
                    "sample_id": ids,
                    "official_fold": fold,
                    "true_label": y_true,
                    "clipped_prediction": values,
                }
            )
        )
        manifest.append(
            {
                "model": model_label,
                "official_fold": fold,
                "source_type": "legacy_clipped_npy_bound_to_official_test_order",
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return pd.concat(frames, ignore_index=True), manifest


def load_l3_npz_predictions(
    results_dir: Path,
    fold_metadata: dict[int, dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    """Load Level-3 clipped predictions and align their explicit IDs."""
    frames = []
    manifest = []
    for fold in EXPECTED_FOLDS:
        path = results_dir / f"fold_{fold}" / "test_preds_clipped.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Level 3 frozen clipped prediction file is missing: {path}"
            )

        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            expected_keys = {"ids", "preds"}
            if keys != expected_keys:
                raise AssertionError(
                    f"Level 3 fold {fold} NPZ keys are {sorted(keys)}; "
                    f"expected exactly {sorted(expected_keys)}."
                )
            raw_ids = np.asarray(archive["ids"])
            values = np.asarray(archive["preds"], dtype=float)

        if raw_ids.ndim != 1:
            raise AssertionError(
                f"Level 3 fold {fold} IDs have shape {raw_ids.shape}; "
                "expected one dimension."
            )
        if raw_ids.dtype.kind not in {"U", "S"}:
            raise AssertionError(
                f"Level 3 fold {fold} IDs have dtype {raw_ids.dtype}; "
                "expected a NumPy string array."
            )
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_ids.tolist()
        ]
        if any(not sample_id for sample_id in ids):
            raise AssertionError(f"Level 3 fold {fold} contains an empty sample ID.")
        if len(ids) != len(set(ids)):
            raise AssertionError(
                f"Level 3 fold {fold} contains duplicate sample IDs."
            )
        if values.ndim != 1 or len(values) != len(ids):
            raise AssertionError(
                f"Level 3 fold {fold} has ID shape {raw_ids.shape} and "
                f"prediction shape {values.shape}; expected equal 1-D lengths."
            )
        assert_clipped_predictions(values, f"Level 3 fold {fold}")

        expected_ids = fold_metadata[fold]["test_ids"]
        expected_set = set(expected_ids)
        observed_set = set(ids)
        if observed_set != expected_set:
            missing_ids = sorted(expected_set - observed_set)[:10]
            extra_ids = sorted(observed_set - expected_set)[:10]
            raise AssertionError(
                f"Level 3 fold {fold} sample IDs do not match the official "
                f"test fold. Missing examples={missing_ids}; "
                f"extra examples={extra_ids}."
            )

        prediction_by_id = dict(zip(ids, values))
        ordered_values = np.asarray(
            [prediction_by_id[sample_id] for sample_id in expected_ids],
            dtype=float,
        )
        frames.append(
            pd.DataFrame(
                {
                    "sample_id": expected_ids,
                    "official_fold": fold,
                    "true_label": fold_metadata[fold]["test_y"],
                    "clipped_prediction": ordered_values,
                }
            )
        )
        manifest.append(
            {
                "model": "Level 3",
                "official_fold": fold,
                "source_type": "clipped_npz_aligned_by_explicit_sample_id",
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "npz_keys": "ids,preds",
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise AssertionError(
            "Level 3 sample IDs are not unique across the five official folds."
        )
    return combined, manifest


def locate_l1bc_predictions(results_dir: Path) -> tuple[str, list[Path]]:
    combined = results_dir / L1BC_PREDICTIONS_NAME
    if combined.is_file():
        return "combined_csv", [combined]

    fold_paths = [
        results_dir / "folds" / f"fold_{fold}_predictions.csv"
        for fold in EXPECTED_FOLDS
    ]
    if all(path.is_file() for path in fold_paths):
        return "per_fold_csv", fold_paths

    missing = [str(path) for path in [combined, *fold_paths] if not path.is_file()]
    formatted = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        "Level 1 + BandCenter prediction CSV was not found. Checked the combined "
        f"file and all per-fold files:\n{formatted}"
    )


def load_l1bc_predictions(
    results_dir: Path,
    fold_metadata: dict[int, dict[str, Any]],
    official_assignment_hash: str,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any] | None]:
    mode, paths = locate_l1bc_predictions(results_dir)
    frames = []
    manifest = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"sample_id": str})
        frames.append(frame)
        manifest.append(
            {
                "model": "Level 1 + BandCenter",
                "official_fold": "all" if mode == "combined_csv" else "from_filename",
                "source_type": mode,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    raw = pd.concat(frames, ignore_index=True)

    required_columns = {
        "sample_id",
        "official_fold",
        "true_label",
        "clipped_prediction",
    }
    missing_columns = sorted(required_columns - set(raw.columns))
    if missing_columns:
        raise AssertionError(
            "Level 1 + BandCenter predictions are missing columns: "
            f"{missing_columns}"
        )
    if raw.duplicated(["sample_id", "official_fold"]).any():
        raise AssertionError(
            "Level 1 + BandCenter predictions contain duplicate sample/fold rows."
        )

    ordered_frames = []
    for fold in EXPECTED_FOLDS:
        expected_ids = fold_metadata[fold]["test_ids"]
        expected_y = fold_metadata[fold]["test_y"]
        part = raw.loc[raw["official_fold"].astype(int) == fold].copy()
        if set(part["sample_id"]) != set(expected_ids):
            missing_ids = sorted(set(expected_ids) - set(part["sample_id"]))[:10]
            extra_ids = sorted(set(part["sample_id"]) - set(expected_ids))[:10]
            raise AssertionError(
                f"Level 1 + BandCenter fold {fold} sample IDs do not match "
                f"the official test fold. Missing examples={missing_ids}; "
                f"extra examples={extra_ids}."
            )
        part = part.set_index("sample_id").loc[expected_ids].reset_index()
        observed_y = part["true_label"].to_numpy(dtype=float)
        if not np.allclose(observed_y, expected_y, rtol=0.0, atol=1e-12):
            maximum_difference = float(np.max(np.abs(observed_y - expected_y)))
            raise AssertionError(
                f"Level 1 + BandCenter fold {fold} labels differ from MatBench "
                f"(maximum absolute difference={maximum_difference})."
            )
        prediction = part["clipped_prediction"].to_numpy(dtype=float)
        assert_clipped_predictions(
            prediction, f"Level 1 + BandCenter fold {fold}"
        )
        ordered_frames.append(
            pd.DataFrame(
                {
                    "sample_id": expected_ids,
                    "official_fold": fold,
                    "true_label": expected_y,
                    "clipped_prediction": prediction,
                }
            )
        )

    training_config = None
    config_path = results_dir / L1BC_CONFIG_NAME
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            training_config = json.load(handle)
        manifest.append(
            {
                "model": "Level 1 + BandCenter",
                "official_fold": "all",
                "source_type": "training_config",
                "path": str(config_path.resolve()),
                "size_bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            }
        )
        if training_config.get("task_name") != TASK_NAME:
            raise AssertionError(
                "Level 1 + BandCenter training config task does not match "
                f"{TASK_NAME}."
            )
        feature_count = training_config.get("feature_counts", {}).get("total")
        if int(feature_count) != 133:
            raise AssertionError(
                "Level 1 + BandCenter training config does not report 133 features."
            )
        saved_hash = training_config.get("fold_assignment_sha256")
        if saved_hash != official_assignment_hash:
            raise AssertionError(
                "Level 1 + BandCenter training fold-assignment hash does not "
                "match the current official MatBench mapping."
            )

    return pd.concat(ordered_frames, ignore_index=True), manifest, training_config


def align_prediction_frames(
    assignments,
    model_frames: dict[str, Any],
) -> Any:
    aligned = assignments.copy()
    expected_keys = set(
        zip(aligned["sample_id"], aligned["official_fold"].astype(int))
    )

    for spec in MODEL_SPECS:
        frame = model_frames[spec["key"]].copy()
        frame["official_fold"] = frame["official_fold"].astype(int)
        observed_keys = set(zip(frame["sample_id"], frame["official_fold"]))
        if observed_keys != expected_keys:
            raise AssertionError(
                f"{spec['label']} prediction keys do not match official assignments."
            )
        lookup = frame.set_index(["sample_id", "official_fold"])
        ordered_index = pd.MultiIndex.from_frame(
            aligned[["sample_id", "official_fold"]]
        )
        ordered = lookup.loc[ordered_index]

        observed_y = ordered["true_label"].to_numpy(dtype=float)
        expected_y = aligned["true_label"].to_numpy(dtype=float)
        if not np.allclose(observed_y, expected_y, rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"{spec['label']} true labels do not match official labels."
            )
        aligned[spec["prediction_column"]] = ordered[
            "clipped_prediction"
        ].to_numpy(dtype=float)

    if aligned.duplicated(["sample_id", "official_fold"]).any():
        raise AssertionError("Aligned predictions contain duplicate sample/fold rows.")
    if aligned[list(spec["prediction_column"] for spec in MODEL_SPECS)].isna().any().any():
        raise AssertionError("Aligned prediction table contains missing predictions.")
    return aligned


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def calculate_per_fold_metrics(aligned) -> Any:
    rows = []
    for fold in EXPECTED_FOLDS:
        part = aligned.loc[aligned["official_fold"] == fold]
        y_true = part["true_label"].to_numpy(dtype=float)
        for spec in MODEL_SPECS:
            prediction = part[spec["prediction_column"]].to_numpy(dtype=float)
            metrics = calculate_metrics(y_true, prediction)
            rows.append(
                {
                    "model": spec["label"],
                    "features": spec["features"],
                    "official_fold": fold,
                    "n_samples": len(part),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                }
            )
    return pd.DataFrame(rows)


def calculate_summary(per_fold) -> Any:
    rows = []
    for spec in MODEL_SPECS:
        part = per_fold.loc[per_fold["model"] == spec["label"]].sort_values(
            "official_fold"
        )
        if part["official_fold"].tolist() != list(EXPECTED_FOLDS):
            raise AssertionError(f"{spec['label']} does not have all five fold metrics.")
        row = {
            "model": spec["label"],
            "features": spec["features"],
        }
        for metric in ("mae", "rmse", "r2"):
            values = part[metric].to_numpy(dtype=float)
            mean_value = float(np.mean(values))
            sd_value = float(np.std(values, ddof=0))
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_fold_sd_ddof_0"] = sd_value
            row[f"{metric}_mean_plus_minus_fold_sd"] = (
                f"{mean_value:.6f} ± {sd_value:.6f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def metric_lookup(per_fold, model: str, fold: int, metric: str) -> float:
    values = per_fold.loc[
        (per_fold["model"] == model) & (per_fold["official_fold"] == fold),
        metric,
    ]
    if len(values) != 1:
        raise AssertionError(
            f"Expected one {metric} value for {model}, fold {fold}; found {len(values)}."
        )
    return float(values.iloc[0])


def calculate_paired_differences(per_fold) -> tuple[Any, Any]:
    rows = []
    for fold in EXPECTED_FOLDS:
        l1 = metric_lookup(per_fold, "Level 1", fold, "mae")
        l1bc = metric_lookup(per_fold, "Level 1 + BandCenter", fold, "mae")
        l2 = metric_lookup(per_fold, "Level 2", fold, "mae")
        l3 = metric_lookup(per_fold, "Level 3", fold, "mae")
        total_l1_l2 = l1 - l2
        rows.append(
            {
                "official_fold": fold,
                "mae_level1": l1,
                "mae_level1_plus_bandcenter": l1bc,
                "mae_level2": l2,
                "mae_level3": l3,
                "delta_bc_l1_minus_l1bc": l1 - l1bc,
                "delta_remaining_l1bc_minus_l2": l1bc - l2,
                "delta_l2_minus_l3": l2 - l3,
                "delta_total_l1_minus_l2": total_l1_l2,
                "delta_total_l1_minus_l3": l1 - l3,
                "sequential_bc_fraction_fold": (
                    (l1 - l1bc) / total_l1_l2
                    if total_l1_l2 != 0
                    else np.nan
                ),
            }
        )
    paired = pd.DataFrame(rows)

    definitions = (
        (
            "delta_bc_l1_minus_l1bc",
            "MAE(Level 1) - MAE(Level 1 + BandCenter)",
            "positive means adding BandCenter improved MAE",
        ),
        (
            "delta_remaining_l1bc_minus_l2",
            "MAE(Level 1 + BandCenter) - MAE(Level 2)",
            "positive means Level 2 retained a further MAE improvement",
        ),
        (
            "delta_l2_minus_l3",
            "MAE(Level 2) - MAE(Level 3)",
            "positive means Level 3 improved over Level 2",
        ),
        (
            "delta_total_l1_minus_l2",
            "MAE(Level 1) - MAE(Level 2)",
            "positive means Level 2 improved over Level 1",
        ),
        (
            "delta_total_l1_minus_l3",
            "MAE(Level 1) - MAE(Level 3)",
            "positive means Level 3 improved over Level 1",
        ),
    )
    summary_rows = []
    for column, formula, interpretation in definitions:
        values = paired[column].to_numpy(dtype=float)
        if np.all(values > 0):
            direction = "all_positive"
            consistent = True
        elif np.all(values < 0):
            direction = "all_negative"
            consistent = True
        elif np.all(values == 0):
            direction = "all_zero"
            consistent = True
        else:
            direction = "mixed_or_contains_zero"
            consistent = False
        summary_rows.append(
            {
                "difference": column,
                "formula": formula,
                "interpretation": interpretation,
                "fold_0": values[0],
                "fold_1": values[1],
                "fold_2": values[2],
                "fold_3": values[3],
                "fold_4": values[4],
                "mean_difference": float(np.mean(values)),
                "difference_fold_sd_ddof_0": float(np.std(values, ddof=0)),
                "all_fold_directions_consistent": consistent,
                "direction": direction,
            }
        )
    return paired, pd.DataFrame(summary_rows)


def summary_mae(summary, model: str) -> float:
    values = summary.loc[summary["model"] == model, "mae_mean"]
    if len(values) != 1:
        raise AssertionError(f"Could not find one summary MAE for {model}.")
    return float(values.iloc[0])


def calculate_sequential_gains(summary) -> Any:
    mae_l1 = summary_mae(summary, "Level 1")
    mae_l1bc = summary_mae(summary, "Level 1 + BandCenter")
    mae_l2 = summary_mae(summary, "Level 2")
    mae_l3 = summary_mae(summary, "Level 3")
    delta_bc = mae_l1 - mae_l1bc
    delta_remaining = mae_l1bc - mae_l2
    delta_l2_l3 = mae_l2 - mae_l3
    total_l1_l2 = mae_l1 - mae_l2
    fraction = delta_bc / total_l1_l2 if total_l1_l2 != 0 else np.nan
    return pd.DataFrame(
        [
            {
                "mae_level1_fold_mean": mae_l1,
                "mae_level1_plus_bandcenter_fold_mean": mae_l1bc,
                "mae_level2_fold_mean": mae_l2,
                "mae_level3_fold_mean": mae_l3,
                "delta_bc_mae_l1_minus_l1bc": delta_bc,
                "delta_remaining_mae_l1bc_minus_l2": delta_remaining,
                "delta_l2_minus_l3": delta_l2_l3,
                "delta_total_mae_l1_minus_l2": total_l1_l2,
                "delta_total_mae_l1_minus_l3": mae_l1 - mae_l3,
                "sequential_bc_fraction": fraction,
                "interpretation": (
                    "BandCenter fraction uses the declared L1 -> L1+BC -> L2 "
                    "feature-addition order; Level 3 is reported separately; "
                    "none is a strict causal contribution"
                ),
            }
        ]
    )


def calculate_subset_metrics(aligned) -> Any:
    y_true = aligned["true_label"].to_numpy(dtype=float)
    masks = (
        ("full_set", np.ones(len(aligned), dtype=bool), "all official test samples"),
        ("zero_gap", y_true == 0.0, "true_label == 0.0 exactly"),
        ("nonzero_gap", y_true != 0.0, "true_label != 0.0 exactly"),
    )
    rows = []
    for subset, mask, definition in masks:
        count = int(mask.sum())
        if count == 0:
            raise AssertionError(f"Subset {subset} contains no samples.")
        for spec in MODEL_SPECS:
            prediction = aligned[spec["prediction_column"]].to_numpy(dtype=float)
            rows.append(
                {
                    "subset": subset,
                    "subset_definition": definition,
                    "model": spec["label"],
                    "features": spec["features"],
                    "n_samples": count,
                    "clipped_mae": float(
                        mean_absolute_error(y_true[mask], prediction[mask])
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_conclusions(sequential, paired_summary) -> dict[str, str]:
    row = sequential.iloc[0]
    delta_bc = float(row["delta_bc_mae_l1_minus_l1bc"])
    total = float(row["delta_total_mae_l1_minus_l2"])
    fraction = float(row["sequential_bc_fraction"])

    if total <= 0 or not np.isfinite(fraction):
        major = (
            "The frozen Level-1 to Level-2 mean-MAE change is not a positive "
            "denominator, so whether BandCenter explains a majority is undefined."
        )
    elif fraction >= 0.5:
        major = (
            "Yes under this descriptive order: adding BandCenter alone accounts "
            f"for {fraction:.1%} of the observed Level-1 to Level-2 mean-MAE gain, "
            "which is a majority by the predeclared 50% descriptive threshold."
        )
    else:
        major = (
            "No under the predeclared 50% descriptive threshold: adding "
            f"BandCenter alone accounts for {fraction:.1%} of the observed "
            "Level-1 to Level-2 mean-MAE gain."
        )

    bc_direction_row = paired_summary.loc[
        paired_summary["difference"] == "delta_bc_l1_minus_l1bc"
    ].iloc[0]
    if delta_bc > 0:
        wording = (
            "Use “BandCenter-plus-structural-descriptor gain” for the Level-1 "
            "to Level-2 change. BandCenter produced a positive sequential "
            "mean-MAE gain, so calling the entire change a structural-descriptor "
            "gain would conflate a composition feature with structural features."
        )
    else:
        wording = (
            "“Structural-descriptor gain” can be retained descriptively because "
            "BandCenter did not produce a positive sequential mean-MAE gain, but "
            "the methods should still state that Level 2 contains BandCenter."
        )

    direction = (
        "All five folds had the same BandCenter-difference direction."
        if bool(bc_direction_row["all_fold_directions_consistent"])
        else "The BandCenter-difference direction was not consistent across all five folds."
    )
    return {
        "major_improvement": major,
        "paper_wording": wording,
        "fold_direction": direction,
        "causal_boundary": (
            "These are sequential gains under this feature-addition order, not "
            "strict causal contributions; BandCenter may interact with or be "
            "redundant with structural descriptors."
        ),
    }


def markdown_summary_table(summary) -> str:
    lines = [
        "| Model | Features | MAE mean ± fold SD | RMSE mean ± fold SD | R² mean ± fold SD |",
        "|:---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.features} | "
            f"{row.mae_mean:.6f} ± {row.mae_fold_sd_ddof_0:.6f} | "
            f"{row.rmse_mean:.6f} ± {row.rmse_fold_sd_ddof_0:.6f} | "
            f"{row.r2_mean:.6f} ± {row.r2_fold_sd_ddof_0:.6f} |"
        )
    return "\n".join(lines)


def markdown_per_fold_table(per_fold) -> str:
    lines = [
        "| Fold | Model | Features | MAE | RMSE | R² |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    model_order = {
        spec["label"]: position for position, spec in enumerate(MODEL_SPECS)
    }
    ordered = per_fold.assign(
        _model_order=per_fold["model"].map(model_order)
    ).sort_values(["official_fold", "_model_order"], kind="stable")
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {int(row.official_fold)} | {row.model} | {row.features} | "
            f"{row.mae:.6f} | {row.rmse:.6f} | {row.r2:.6f} |"
        )
    return "\n".join(lines)


def markdown_paired_table(paired) -> str:
    lines = [
        "| Fold | ΔBC: L1 − L1+BC | Δremaining: L1+BC − L2 | ΔL2→L3: L2 − L3 | Δtotal: L1 − L2 | Δtotal: L1 − L3 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {int(row.official_fold)} | "
            f"{row.delta_bc_l1_minus_l1bc:.6f} | "
            f"{row.delta_remaining_l1bc_minus_l2:.6f} | "
            f"{row.delta_l2_minus_l3:.6f} | "
            f"{row.delta_total_l1_minus_l2:.6f} | "
            f"{row.delta_total_l1_minus_l3:.6f} |"
        )
    return "\n".join(lines)


def make_audit(
    output_dir: Path,
    resolved_dirs: dict[str, Path],
    assignment_hash: str,
    training_config: dict[str, Any] | None,
    summary,
    per_fold,
    paired,
    paired_summary,
    sequential,
    subset_metrics,
    conclusions: dict[str, str],
) -> None:
    gain = sequential.iloc[0]
    paired_bc = paired_summary.loc[
        paired_summary["difference"] == "delta_bc_l1_minus_l1bc"
    ].iloc[0]
    paired_remaining = paired_summary.loc[
        paired_summary["difference"] == "delta_remaining_l1bc_minus_l2"
    ].iloc[0]
    paired_l2_l3 = paired_summary.loc[
        paired_summary["difference"] == "delta_l2_minus_l3"
    ].iloc[0]

    subset_lines = [
        "| Subset | Model | N | Clipped MAE |",
        "|:---|:---|---:|---:|",
    ]
    for row in subset_metrics.itertuples(index=False):
        subset_lines.append(
            f"| {row.subset} | {row.model} | {int(row.n_samples)} | "
            f"{row.clipped_mae:.6f} |"
        )

    config_note = (
        "The Level-1+BandCenter training config was loaded and its task, "
        "133-feature count, and official fold-assignment hash were verified."
        if training_config is not None
        else (
            "No Level-1+BandCenter training config was found; prediction IDs, "
            "folds, labels, and non-negativity were still verified directly."
        )
    )

    text = f"""# Level 1 / BandCenter / Level 2 / Level 3 comparison audit

## Sources

- Level 1 frozen results: `{resolved_dirs["l1"].resolve()}`
- Level 1 + BandCenter results: `{resolved_dirs["l1bc"].resolve()}`
- Level 2 frozen results: `{resolved_dirs["l2"].resolve()}`
- Level 3 frozen results: `{resolved_dirs["l3"].resolve()}`
- Source file paths and SHA-256 hashes: `{SOURCE_MANIFEST_NAME}`
- No model was retrained and no feature was generated by this script.

The legacy Level-1 and Level-2 `pred_fold_{{fold}}.npy` files contain no sample
IDs, so each array was bound to the ordered official MatBench test IDs using the
same positional convention as the uploaded frozen scripts. Array lengths were
required to match each official fold exactly. Level-1+BandCenter predictions
were aligned by explicit sample ID and fold. Each Level-3
`fold_{{fold}}/test_preds_clipped.npz` was required to contain exactly the
`ids` and `preds` arrays; its predictions were reordered by explicit sample ID
to the official test-fold order.

{config_note}

## Alignment assertions

- MatBench task: `{TASK_NAME}`
- Official folds: `{list(EXPECTED_FOLDS)}`
- Every official test sample appeared exactly once across the five folds.
- All four models had exactly one finite, nonnegative clipped prediction for
  every official sample.
- All true labels agreed with MatBench.
- Official fold-assignment SHA-256: `{assignment_hash}`
- The fully aligned prediction table is `{ALIGNED_PREDICTIONS_NAME}`.

## Five-fold summary

Fold SD uses the population convention `ddof=0`, matching the declared
MatBench/paper table convention. All metrics use clipped predictions.

{markdown_summary_table(summary)}

## Per-fold results

{markdown_per_fold_table(per_fold)}

## Sequential gains

Using the five-fold mean MAEs:

- ΔBC = MAE(L1) − MAE(L1+BC) =
  **{gain["delta_bc_mae_l1_minus_l1bc"]:.6f} eV**
- Δremaining = MAE(L1+BC) − MAE(L2) =
  **{gain["delta_remaining_mae_l1bc_minus_l2"]:.6f} eV**
- Total Level-1 to Level-2 gain =
  **{gain["delta_total_mae_l1_minus_l2"]:.6f} eV**
- Level-2 to Level-3 MAE change, MAE(L2) − MAE(L3) =
  **{gain["delta_l2_minus_l3"]:.6f} eV**
- Total Level-1 to Level-3 MAE change =
  **{gain["delta_total_mae_l1_minus_l3"]:.6f} eV**
- Sequential BC fraction =
  **{gain["sequential_bc_fraction"]:.6%}**

This fraction is a **sequential gain under this feature-addition order**, not a
strict causal contribution of BandCenter.

## Paired fold MAE differences

{markdown_paired_table(paired)}

- ΔBC mean ± fold SD:
  **{paired_bc["mean_difference"]:.6f} ± {paired_bc["difference_fold_sd_ddof_0"]:.6f} eV**
- ΔBC all-fold direction consistent:
  **{bool(paired_bc["all_fold_directions_consistent"])}**
- Δremaining mean ± fold SD:
  **{paired_remaining["mean_difference"]:.6f} ± {paired_remaining["difference_fold_sd_ddof_0"]:.6f} eV**
- Δremaining all-fold direction consistent:
  **{bool(paired_remaining["all_fold_directions_consistent"])}**
- ΔL2→L3 mean ± fold SD:
  **{paired_l2_l3["mean_difference"]:.6f} ± {paired_l2_l3["difference_fold_sd_ddof_0"]:.6f} eV**
- ΔL2→L3 all-fold direction consistent:
  **{bool(paired_l2_l3["all_fold_directions_consistent"])}**

## Low-cost subsets

Zero gap is defined exactly as `true_label == 0.0`.

{chr(10).join(subset_lines)}

## Interpretation

- **Does BandCenter alone explain the major improvement?**
  {conclusions["major_improvement"]}
- **Fold consistency:** {conclusions["fold_direction"]}
- **Recommended paper wording:** {conclusions["paper_wording"]}
- **Boundary:** {conclusions["causal_boundary"]}
"""
    atomic_write_text(text, output_dir / AUDIT_NAME)


def run(args: argparse.Namespace) -> None:
    l1_requested = args.l1_results_dir.expanduser()
    l1bc_requested = args.l1bc_results_dir.expanduser()
    l2_requested = args.l2_results_dir.expanduser()
    l3_requested = args.l3_results_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    prepare_output_dir(output_dir)
    log_path = output_dir / LOG_NAME
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print("=" * 78)
            print(
                "Frozen Level 1 / Level 1 + BandCenter / Level 2 / "
                "Level 3 comparison"
            )
            print("=" * 78)
            print(f"Started UTC: {utc_now()}")
            print(f"Output directory: {output_dir.resolve()}")

            load_runtime_dependencies()
            versions = collect_versions()
            print("Runtime versions:")
            for name, version in versions.items():
                print(f"  {name}: {version}")

            resolved_dirs = {
                "l1": resolve_input_dir(l1_requested, "Level 1"),
                "l1bc": resolve_input_dir(
                    l1bc_requested, "Level 1 + BandCenter"
                ),
                "l2": resolve_input_dir(
                    l2_requested,
                    "Level 2",
                    fallback=(
                        DEFAULT_L2_FALLBACK_DIR
                        if l2_requested == DEFAULT_L2_RESULTS_DIR
                        else None
                    ),
                ),
                "l3": resolve_input_dir(l3_requested, "Level 3"),
            }
            for key, path in resolved_dirs.items():
                print(f"Resolved {key} results: {path.resolve()}")

            task = get_task()
            fold_metadata, assignments, assignment_hash = (
                load_official_fold_metadata(task)
            )

            l1_frame, l1_manifest = load_legacy_npy_predictions(
                resolved_dirs["l1"], "Level 1", fold_metadata
            )
            l1bc_frame, l1bc_manifest, training_config = load_l1bc_predictions(
                resolved_dirs["l1bc"], fold_metadata, assignment_hash
            )
            l2_frame, l2_manifest = load_legacy_npy_predictions(
                resolved_dirs["l2"], "Level 2", fold_metadata
            )
            l3_frame, l3_manifest = load_l3_npz_predictions(
                resolved_dirs["l3"], fold_metadata
            )
            manifest = pd.DataFrame(
                [*l1_manifest, *l1bc_manifest, *l2_manifest, *l3_manifest]
            )

            aligned = align_prediction_frames(
                assignments,
                {
                    "l1": l1_frame,
                    "l1bc": l1bc_frame,
                    "l2": l2_frame,
                    "l3": l3_frame,
                },
            )
            per_fold = calculate_per_fold_metrics(aligned)
            summary = calculate_summary(per_fold)
            paired, paired_summary = calculate_paired_differences(per_fold)
            sequential = calculate_sequential_gains(summary)
            subset_metrics = calculate_subset_metrics(aligned)
            conclusions = build_conclusions(sequential, paired_summary)

            atomic_write_csv(
                assignments[
                    [
                        "sample_id",
                        "official_fold",
                        "official_test_position",
                        "true_label",
                    ]
                ],
                output_dir / FOLD_ASSIGNMENTS_NAME,
            )
            atomic_write_csv(aligned, output_dir / ALIGNED_PREDICTIONS_NAME)
            atomic_write_csv(per_fold, output_dir / PER_FOLD_NAME)
            atomic_write_csv(summary, output_dir / SUMMARY_NAME)
            atomic_write_csv(paired, output_dir / PAIRED_DIFFERENCES_NAME)
            atomic_write_csv(paired_summary, output_dir / PAIRED_SUMMARY_NAME)
            atomic_write_csv(sequential, output_dir / SEQUENTIAL_GAINS_NAME)
            atomic_write_csv(subset_metrics, output_dir / SUBSET_METRICS_NAME)
            atomic_write_csv(manifest, output_dir / SOURCE_MANIFEST_NAME)

            config = {
                "comparison_name": (
                    "Level 1 / Level 1 + BandCenter / Level 2 / Level 3"
                ),
                "task_name": TASK_NAME,
                "created_utc": utc_now(),
                "requested_paths": {
                    "level1": str(l1_requested),
                    "level1_plus_bandcenter": str(l1bc_requested),
                    "level2": str(l2_requested),
                    "level3": str(l3_requested),
                    "output": str(output_dir),
                },
                "resolved_paths": {
                    "level1": str(resolved_dirs["l1"].resolve()),
                    "level1_plus_bandcenter": str(
                        resolved_dirs["l1bc"].resolve()
                    ),
                    "level2": str(resolved_dirs["l2"].resolve()),
                    "level3": str(resolved_dirs["l3"].resolve()),
                    "output": str(output_dir.resolve()),
                },
                "models": list(MODEL_SPECS),
                "prediction_policy": {
                    "level1": (
                        "frozen clipped .npy bound to ordered official test IDs"
                    ),
                    "level1_plus_bandcenter": (
                        "clipped CSV aligned by explicit sample ID and fold"
                    ),
                    "level2": (
                        "frozen clipped .npy bound to ordered official test IDs"
                    ),
                    "level3": (
                        "frozen fold NPZ (ids,preds), strictly aligned by "
                        "explicit sample ID"
                    ),
                },
                "metric_policy": {
                    "all_metrics_use": "non-negativity-clipped predictions",
                    "aggregate": "mean of five official fold metrics",
                    "fold_sd_ddof": 0,
                    "zero_gap_definition": "true_label == 0.0 exactly",
                },
                "sequential_gain_definition": {
                    "delta_bc": "MAE(Level 1) - MAE(Level 1 + BandCenter)",
                    "delta_remaining": (
                        "MAE(Level 1 + BandCenter) - MAE(Level 2)"
                    ),
                    "delta_l2_to_l3": "MAE(Level 2) - MAE(Level 3)",
                    "fraction": (
                        "(MAE(Level 1) - MAE(Level 1 + BandCenter)) / "
                        "(MAE(Level 1) - MAE(Level 2))"
                    ),
                    "interpretation": (
                        "sequential gain under this feature-addition order; "
                        "not strict causal contribution"
                    ),
                    "majority_descriptive_threshold": 0.5,
                },
                "official_folds": list(EXPECTED_FOLDS),
                "fold_assignment_sha256": assignment_hash,
                "source_manifest_sha256": sha256_text(
                    manifest.to_csv(index=False)
                ),
                "versions": versions,
            }
            atomic_write_json(config, output_dir / CONFIG_NAME)
            make_audit(
                output_dir,
                resolved_dirs,
                assignment_hash,
                training_config,
                summary,
                per_fold,
                paired,
                paired_summary,
                sequential,
                subset_metrics,
                conclusions,
            )

            print("\n" + markdown_summary_table(summary))
            gain = sequential.iloc[0]
            print("\nSequential gains:")
            print(
                "  ΔBC = "
                f"{gain['delta_bc_mae_l1_minus_l1bc']:.6f} eV"
            )
            print(
                "  Δremaining = "
                f"{gain['delta_remaining_mae_l1bc_minus_l2']:.6f} eV"
            )
            print(
                "  ΔL2→L3 = "
                f"{gain['delta_l2_minus_l3']:.6f} eV"
            )
            print(
                "  sequential BC fraction = "
                f"{gain['sequential_bc_fraction']:.6%}"
            )
            print(f"\n{conclusions['major_improvement']}")
            print(conclusions["paper_wording"])
            print(conclusions["causal_boundary"])
            print(f"\nSummary CSV: {output_dir / SUMMARY_NAME}")
            print(f"Audit: {output_dir / AUDIT_NAME}")
            print(f"Finished UTC: {utc_now()}")
            print("=" * 78)
        except Exception:
            print("\nFATAL ERROR", file=sys.stderr)
            traceback.print_exc()
            print(
                "\nNo source prediction or training-result files were modified.",
                file=sys.stderr,
            )
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
