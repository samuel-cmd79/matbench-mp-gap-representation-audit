#!/usr/bin/env python3
"""
Train the MatBench mp-gap Level-1 + BandCenter XGBoost control.

This script is intentionally independent from the original baseline/ablation
scripts. It reads only the frozen, pre-imputation composition-feature caches:

    fold_{0..4}_{train|test}_ElementProperty.pkl
    fold_{0..4}_{train|test}_BandCenter.pkl

It never instantiates a matminer featurizer and never regenerates a missing
cache. The imputation order deliberately reproduces the uploaded frozen
baseline/ablation scripts:

    1. Compute column means on the full official outer train-and-validation fold.
    2. Fill the full outer train-and-validation fold and official test fold.
    3. Split the already-imputed outer train-and-validation fold 80/20 for
       XGBoost fitting and early-stopping validation.

The separate Level-1 / Level-2 comparison is intentionally not performed here.

Examples
--------
Run with the project-relative defaults:

    python train_l1_plus_bandcenter.py

Choose explicit paths:

    python train_l1_plus_bandcenter.py \
        --cache-dir ../matbench_cache \
        --output-dir ../matbench_outputs_l1_plus_bandcenter_run0731

Resume after an interruption:

    python train_l1_plus_bandcenter.py --resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_NAME = "matbench_mp_gap"
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_LEVEL1_FEATURES = 132
EXPECTED_BANDCENTER_FEATURES = 1
EXPECTED_TOTAL_FEATURES = 133

SEED = 42
VALIDATION_FRACTION = 0.20
MAX_BOOST_ROUNDS = 8_000
EARLY_STOPPING_PATIENCE = 200

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "eta": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 1e-3,
    "tree_method": "hist",
    "seed": SEED,
}

DEFAULT_CACHE_DIR = Path("../matbench_cache")
DEFAULT_OUTPUT_DIR = Path("../matbench_outputs_l1_plus_bandcenter_run0731")

PREDICTIONS_NAME = "l1_plus_bandcenter_predictions.csv"
FOLD_METRICS_NAME = "l1_plus_bandcenter_fold_metrics.csv"
SUMMARY_METRICS_NAME = "l1_plus_bandcenter_summary_metrics.csv"
CONFIG_NAME = "l1_plus_bandcenter_config.json"
FEATURE_LIST_NAME = "l1_plus_bandcenter_feature_list.csv"
FOLD_ASSIGNMENTS_NAME = "l1_plus_bandcenter_fold_assignments.csv"
INTERNAL_SPLIT_NAME = "l1_plus_bandcenter_internal_split.csv"
CACHE_MANIFEST_NAME = "l1_plus_bandcenter_cache_manifest.csv"
MATBENCH_SCORES_NAME = "l1_plus_bandcenter_matbench_scores.json"
AUDIT_NAME = "l1_plus_bandcenter_audit.md"
LOG_NAME = "l1_plus_bandcenter_run.log"

FORBIDDEN_FAMILY_NAMES = (
    "DensityFeatures",
    "GlobalSymmetryFeatures",
    "StructuralHeterogeneity",
    "ChemicalOrdering",
    "Dimensionality",
    "SiteStatsFingerprint",
)


# Loaded only after argparse has handled --help, so the help command works even
# in a shell where the scientific environment is not yet activated.
np = None
pd = None
xgb = None
mean_absolute_error = None
mean_squared_error = None
r2_score = None
train_test_split = None
MatbenchBenchmark = None


class TeeStream:
    """Write a stream to both the terminal and a log file."""

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
            "Train the 133-feature MatBench mp-gap Level-1 + BandCenter "
            "XGBoost control using frozen per-featurizer caches."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Frozen feature-cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"New result directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run from completed per-fold prediction and "
            "metric files. Without this flag, a non-empty output directory is refused."
        ),
    )
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    global np, pd, xgb
    global mean_absolute_error, mean_squared_error, r2_score, train_test_split
    global MatbenchBenchmark

    try:
        import numpy as _np
        import pandas as _pd
        import xgboost as _xgb
        from matbench.bench import MatbenchBenchmark as _MatbenchBenchmark
        from sklearn.metrics import (
            mean_absolute_error as _mean_absolute_error,
            mean_squared_error as _mean_squared_error,
            r2_score as _r2_score,
        )
        from sklearn.model_selection import train_test_split as _train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Activate the same Python environment "
            "used for the frozen Level-1/Level-2 runs before executing this script. "
            f"Original import error: {exc}"
        ) from exc

    np = _np
    pd = _pd
    xgb = _xgb
    mean_absolute_error = _mean_absolute_error
    mean_squared_error = _mean_squared_error
    r2_score = _r2_score
    train_test_split = _train_test_split
    MatbenchBenchmark = _MatbenchBenchmark


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(*distribution_names: str) -> str:
    for name in distribution_names:
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
        "xgboost": package_version("xgboost"),
        "matbench": package_version("matbench"),
        "matminer": package_version("matminer"),
        "pymatgen": package_version("pymatgen"),
        "scikit-learn": package_version("scikit-learn", "sklearn"),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
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


def prepare_output_dir(output_dir: Path, resume: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(f"Output path exists but is not a directory: {output_dir}")
        has_contents = any(output_dir.iterdir())
        if has_contents and not resume:
            raise RuntimeError(
                f"Output directory is not empty: {output_dir}\n"
                "Choose a new --output-dir, or pass --resume only if this is an "
                "interrupted run of the same experiment."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "folds").mkdir(parents=True, exist_ok=True)


def cache_path(cache_dir: Path, fold: int, split: str, family: str) -> Path:
    return cache_dir / f"fold_{fold}_{split}_{family}.pkl"


def required_cache_paths(cache_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for fold in EXPECTED_FOLDS:
        for split in ("train", "test"):
            for family in ("ElementProperty", "BandCenter"):
                rows.append(
                    {
                        "official_fold": fold,
                        "split": split,
                        "feature_family": family,
                        "path": cache_path(cache_dir, fold, split, family),
                    }
                )
    return rows


def get_task() -> Any:
    benchmark = MatbenchBenchmark(autoload=False)
    task = next(
        (candidate for candidate in benchmark.tasks if candidate.dataset_name == TASK_NAME),
        None,
    )
    if task is None:
        raise RuntimeError(f"MatBench task not found: {TASK_NAME}")
    task.load()
    folds = tuple(int(v) for v in task.folds)
    if folds != EXPECTED_FOLDS:
        raise AssertionError(
            f"Expected official folds {EXPECTED_FOLDS}, found {folds}."
        )
    return task


def series_ids(series: Any, label: str) -> list[str]:
    if not hasattr(series, "index"):
        raise AssertionError(f"{label} has no pandas index; cannot recover sample IDs.")
    ids = [str(value) for value in series.index.tolist()]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{label} contains duplicate sample IDs.")
    return ids


def labels_to_numpy(labels: Any, ids: list[str], label: str):
    if hasattr(labels, "index"):
        label_ids = [str(value) for value in labels.index.tolist()]
        if label_ids != ids:
            raise AssertionError(f"{label} label index does not match input sample IDs.")
    values = np.asarray(labels, dtype=float)
    if values.ndim != 1 or len(values) != len(ids):
        raise AssertionError(
            f"{label} labels have unexpected shape {values.shape}; expected ({len(ids)},)."
        )
    if not np.isfinite(values).all():
        raise AssertionError(f"{label} labels contain NaN or infinity.")
    return values


def load_official_fold_metadata(task: Any) -> dict[int, dict[str, Any]]:
    metadata = {}
    reference_universe = None
    all_test_ids = []

    for fold in EXPECTED_FOLDS:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

        train_ids = series_ids(train_inputs, f"fold {fold} official train")
        test_ids = series_ids(test_inputs, f"fold {fold} official test")
        train_y = labels_to_numpy(
            train_outputs, train_ids, f"fold {fold} official train"
        )
        test_y = labels_to_numpy(test_outputs, test_ids, f"fold {fold} official test")

        train_set = set(train_ids)
        test_set = set(test_ids)
        if train_set & test_set:
            raise AssertionError(f"Fold {fold} official train and test IDs overlap.")

        universe = train_set | test_set
        if reference_universe is None:
            reference_universe = universe
        elif universe != reference_universe:
            raise AssertionError(
                f"Fold {fold} does not contain the same full sample universe as fold 0."
            )

        all_test_ids.extend(test_ids)
        metadata[fold] = {
            "train_ids": train_ids,
            "test_ids": test_ids,
            "train_y": train_y,
            "test_y": test_y,
        }

    if len(all_test_ids) != len(set(all_test_ids)):
        raise AssertionError("A sample appears in more than one official test fold.")
    if set(all_test_ids) != reference_universe:
        raise AssertionError(
            "The five official test folds do not cover the full sample universe exactly once."
        )

    return metadata


def validate_cache_index(frame, expected_ids: list[str], label: str) -> str:
    index_values = frame.index.tolist()
    if index_values == list(range(len(frame))):
        return "positional_range_index"
    if [str(value) for value in index_values] == expected_ids:
        return "sample_id_index"
    raise AssertionError(
        f"{label} cache index is neither 0..N-1 nor the ordered official sample IDs. "
        "Refusing ambiguous row alignment."
    )


def load_cache_frame(
    path: Path,
    expected_rows: int,
    expected_cols: int,
    expected_ids: list[str],
    label: str,
) -> tuple[Any, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen cache is missing: {path}\n"
            "This script never regenerates missing features."
        )
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise AssertionError(f"{label} is not a pandas DataFrame: {type(frame)!r}")
    if frame.shape != (expected_rows, expected_cols):
        raise AssertionError(
            f"{label} shape is {frame.shape}, expected "
            f"({expected_rows}, {expected_cols})."
        )
    if not frame.columns.is_unique:
        raise AssertionError(f"{label} contains duplicate feature columns.")

    index_mode = validate_cache_index(frame, expected_ids, label)
    frame = frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    if len(frame.columns) != len(set(frame.columns)):
        raise AssertionError(f"{label} columns collide after string conversion.")
    frame = frame.reset_index(drop=True)
    return frame, index_mode


def normalize_numeric(frame):
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def load_fold_features(
    cache_dir: Path,
    fold: int,
    split: str,
    expected_ids: list[str],
) -> tuple[Any, dict[str, Any]]:
    element_path = cache_path(cache_dir, fold, split, "ElementProperty")
    band_path = cache_path(cache_dir, fold, split, "BandCenter")

    element, element_index_mode = load_cache_frame(
        element_path,
        expected_rows=len(expected_ids),
        expected_cols=EXPECTED_LEVEL1_FEATURES,
        expected_ids=expected_ids,
        label=f"fold {fold} {split} ElementProperty",
    )
    band, band_index_mode = load_cache_frame(
        band_path,
        expected_rows=len(expected_ids),
        expected_cols=EXPECTED_BANDCENTER_FEATURES,
        expected_ids=expected_ids,
        label=f"fold {fold} {split} BandCenter",
    )

    overlap = set(element.columns) & set(band.columns)
    if overlap:
        raise AssertionError(
            f"fold {fold} {split} has overlapping ElementProperty/BandCenter "
            f"columns: {sorted(overlap)}"
        )
    non_magpie_columns = [
        column for column in element.columns if not column.startswith("MagpieData ")
    ]
    if non_magpie_columns:
        raise AssertionError(
            f"fold {fold} {split} ElementProperty cache contains columns that "
            "are not frozen MagpieData features: "
            f"{non_magpie_columns[:10]}"
        )

    combined = pd.concat([element, band], axis=1)
    if combined.shape != (len(expected_ids), EXPECTED_TOTAL_FEATURES):
        raise AssertionError(
            f"fold {fold} {split} combined shape is {combined.shape}, expected "
            f"({len(expected_ids)}, {EXPECTED_TOTAL_FEATURES})."
        )
    if not combined.columns.is_unique:
        raise AssertionError(f"fold {fold} {split} combined columns are not unique.")

    for family in FORBIDDEN_FAMILY_NAMES:
        if any(family.lower() in column.lower() for column in combined.columns):
            raise AssertionError(
                f"Forbidden structural family name {family!r} appears in a selected column."
            )

    combined = normalize_numeric(combined)
    band_column = band.columns[0]
    audit = {
        "official_fold": fold,
        "split": split,
        "rows": len(combined),
        "element_property_columns": len(element.columns),
        "bandcenter_columns": len(band.columns),
        "combined_columns": len(combined.columns),
        "bandcenter_column_name": band_column,
        "bandcenter_missing_count": int(combined[band_column].isna().sum()),
        "bandcenter_missing_rate": float(combined[band_column].isna().mean()),
        "element_property_index_mode": element_index_mode,
        "bandcenter_index_mode": band_index_mode,
    }
    return combined, audit


def preflight(
    task: Any,
    cache_dir: Path,
) -> tuple[
    dict[int, dict[str, Any]],
    list[str],
    str,
    Any,
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    print("Running strict preflight checks...")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

    required = required_cache_paths(cache_dir)
    missing = [str(row["path"]) for row in required if not row["path"].is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required frozen cache files are missing. No features were regenerated:\n"
            f"{formatted}"
        )

    fold_metadata = load_official_fold_metadata(task)
    reference_feature_names = None
    reference_band_column = None
    cache_audits = []
    cache_manifest = []

    for row in required:
        path = row["path"]
        cache_manifest.append(
            {
                "official_fold": row["official_fold"],
                "split": row["split"],
                "feature_family": row["feature_family"],
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    for fold in EXPECTED_FOLDS:
        info = fold_metadata[fold]
        train_x, train_audit = load_fold_features(
            cache_dir, fold, "train", info["train_ids"]
        )
        test_x, test_audit = load_fold_features(
            cache_dir, fold, "test", info["test_ids"]
        )
        cache_audits.extend([train_audit, test_audit])

        train_names = list(train_x.columns)
        test_names = list(test_x.columns)
        if train_names != test_names:
            raise AssertionError(
                f"Fold {fold} train/test feature names or ordering differ."
            )
        if reference_feature_names is None:
            reference_feature_names = train_names
            reference_band_column = train_audit["bandcenter_column_name"]
        elif train_names != reference_feature_names:
            raise AssertionError(
                f"Fold {fold} feature names/order differ from fold 0."
            )
        if train_audit["bandcenter_column_name"] != reference_band_column:
            raise AssertionError(
                f"Fold {fold} BandCenter column differs from fold 0."
            )
        if test_audit["bandcenter_column_name"] != reference_band_column:
            raise AssertionError(
                f"Fold {fold} test BandCenter column differs from fold 0."
            )

        del train_x, test_x

    if reference_feature_names is None or reference_band_column is None:
        raise AssertionError("No features were discovered during preflight.")
    if len(reference_feature_names) != EXPECTED_TOTAL_FEATURES:
        raise AssertionError(
            f"Final feature list has {len(reference_feature_names)} columns, "
            f"expected {EXPECTED_TOTAL_FEATURES}."
        )

    assignments = []
    for fold in EXPECTED_FOLDS:
        for sample_id in fold_metadata[fold]["test_ids"]:
            assignments.append({"sample_id": sample_id, "official_fold": fold})
    assignments_df = pd.DataFrame(assignments).sort_values(
        ["sample_id", "official_fold"], kind="stable"
    )
    canonical = "".join(
        f"{row.sample_id},{int(row.official_fold)}\n"
        for row in assignments_df.itertuples(index=False)
    )
    fold_assignment_hash = sha256_text(canonical)

    print(
        "Preflight passed: 132 ElementProperty + 1 BandCenter = 133 features; "
        "all five official folds and all 20 required caches validated."
    )
    return (
        fold_metadata,
        reference_feature_names,
        reference_band_column,
        assignments_df,
        fold_assignment_hash,
        cache_manifest,
        cache_audits,
    )


def feature_list_frame(feature_names: list[str], band_column: str):
    rows = []
    for position, feature in enumerate(feature_names, start=1):
        source = "BandCenter" if feature == band_column else "ElementProperty"
        rows.append(
            {
                "position_1_based": position,
                "feature_name": feature,
                "source_family": source,
            }
        )
    frame = pd.DataFrame(rows)
    if int((frame["source_family"] == "ElementProperty").sum()) != 132:
        raise AssertionError("Feature provenance table does not contain 132 Level-1 rows.")
    if int((frame["source_family"] == "BandCenter").sum()) != 1:
        raise AssertionError("Feature provenance table does not contain one BandCenter row.")
    return frame


def core_run_signature(config: dict[str, Any]) -> str:
    signature_payload = {
        "task_name": config["task_name"],
        "cache_dir": config["cache_dir"],
        "imputation": config["imputation"],
        "internal_split": config["internal_split"],
        "xgboost": config["xgboost"],
        "clipping": config["clipping"],
        "feature_names": config["feature_names"],
        "fold_assignment_sha256": config["fold_assignment_sha256"],
        "cache_manifest": config["cache_manifest"],
        "versions": config["versions"],
    }
    canonical = json.dumps(
        json_safe(signature_payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_text(canonical)


def prepare_config(
    args: argparse.Namespace,
    task: Any,
    feature_names: list[str],
    band_column: str,
    fold_assignment_hash: str,
    cache_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    target = task.metadata.get("target", "unknown")
    config = {
        "experiment_name": "Level 1 + BandCenter",
        "task_name": TASK_NAME,
        "target": target,
        "created_utc": utc_now(),
        "cache_dir": str(args.cache_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "cache_policy": {
            "read_only": True,
            "regenerate_on_miss": False,
            "element_property_template": "fold_{fold}_{split}_ElementProperty.pkl",
            "bandcenter_template": "fold_{fold}_{split}_BandCenter.pkl",
        },
        "feature_counts": {
            "level1_element_property": EXPECTED_LEVEL1_FEATURES,
            "bandcenter": EXPECTED_BANDCENTER_FEATURES,
            "total": EXPECTED_TOTAL_FEATURES,
        },
        "feature_names": feature_names,
        "bandcenter_column_name": band_column,
        "forbidden_structural_families": list(FORBIDDEN_FAMILY_NAMES),
        "imputation": {
            "policy": "uploaded_script_outer_train_fold_column_mean_then_zero_fallback",
            "fit_scope": "full_official_outer_train_and_validation_fold",
            "application": [
                "full_official_outer_train_and_validation_fold",
                "official_test_fold",
            ],
            "order_relative_to_internal_split": "imputation_before_80_20_split",
            "zero_fallback_after_mean": True,
            "missingness_indicator": False,
        },
        "internal_split": {
            "method": "sklearn.model_selection.train_test_split",
            "test_size": VALIDATION_FRACTION,
            "random_state": SEED,
            "shuffle": True,
        },
        "xgboost": {
            "params": XGB_PARAMS,
            "maximum_boosting_rounds": MAX_BOOST_ROUNDS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "validation_metric": "mae",
            "prediction_iteration_policy": "0_through_best_iteration_inclusive",
            "best_iteration_indexing": "zero_based",
        },
        "clipping": {
            "enabled": True,
            "operation": "maximum(raw_prediction, 0.0)",
            "metrics_use": "clipped_prediction",
        },
        "official_folds": list(EXPECTED_FOLDS),
        "fold_assignment_sha256": fold_assignment_hash,
        "fold_sd_ddof": 1,
        "cache_manifest": cache_manifest,
        "versions": collect_versions(),
        "comparison": {
            "performed_in_this_script": False,
            "note": "Level-1/Level-2 comparison is deferred to a separate script.",
        },
    }
    config["run_signature_sha256"] = core_run_signature(config)
    return config


def validate_resume_config(output_dir: Path, config: dict[str, Any], resume: bool) -> None:
    path = output_dir / CONFIG_NAME
    if not path.exists():
        return
    if not resume:
        raise RuntimeError(f"Existing config found without --resume: {path}")
    with path.open("r", encoding="utf-8") as handle:
        previous = json.load(handle)
    previous_signature = previous.get("run_signature_sha256")
    current_signature = config.get("run_signature_sha256")
    if previous_signature != current_signature:
        raise RuntimeError(
            "The existing output directory belongs to a different run signature. "
            "Refusing to mix results. Choose a new --output-dir."
        )
    config["created_utc"] = previous.get("created_utc", config["created_utc"])
    config["last_resumed_utc"] = utc_now()


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def count_missing(values) -> tuple[int, float]:
    count = int(values.isna().sum())
    rate = float(values.isna().mean())
    return count, rate


def predict_best_iteration(booster: Any, dtest: Any, best_iteration: int):
    try:
        return booster.predict(dtest, iteration_range=(0, best_iteration + 1))
    except TypeError:
        return booster.predict(dtest, ntree_limit=best_iteration + 1)


def rebuild_combined_outputs(output_dir: Path) -> None:
    prediction_frames = []
    metric_rows = []
    split_frames = []

    for fold in EXPECTED_FOLDS:
        fold_dir = output_dir / "folds"
        pred_path = fold_dir / f"fold_{fold}_predictions.csv"
        metric_path = fold_dir / f"fold_{fold}_metrics.json"
        split_path = fold_dir / f"fold_{fold}_internal_split.csv"
        if pred_path.exists():
            prediction_frames.append(pd.read_csv(pred_path))
        if metric_path.exists():
            with metric_path.open("r", encoding="utf-8") as handle:
                metric_rows.append(json.load(handle))
        if split_path.exists():
            split_frames.append(pd.read_csv(split_path))

    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions = predictions.sort_values(
            ["official_fold", "sample_id"], kind="stable"
        )
        atomic_write_csv(predictions, output_dir / PREDICTIONS_NAME)

    if metric_rows:
        metrics = pd.DataFrame(metric_rows).sort_values(
            "official_fold", kind="stable"
        )
        atomic_write_csv(metrics, output_dir / FOLD_METRICS_NAME)

    if split_frames:
        splits = pd.concat(split_frames, ignore_index=True)
        splits = splits.sort_values(
            ["official_fold", "outer_train_position"], kind="stable"
        )
        atomic_write_csv(splits, output_dir / INTERNAL_SPLIT_NAME)


def completed_fold_is_valid(
    output_dir: Path,
    fold: int,
    expected_test_ids: list[str],
) -> tuple[bool, Any | None]:
    fold_dir = output_dir / "folds"
    pred_path = fold_dir / f"fold_{fold}_predictions.csv"
    metric_path = fold_dir / f"fold_{fold}_metrics.json"
    split_path = fold_dir / f"fold_{fold}_internal_split.csv"
    if not (pred_path.exists() and metric_path.exists() and split_path.exists()):
        return False, None

    frame = pd.read_csv(pred_path, dtype={"sample_id": str})
    required_columns = {
        "sample_id",
        "official_fold",
        "true_label",
        "raw_prediction",
        "clipped_prediction",
    }
    if not required_columns.issubset(frame.columns):
        return False, None
    if frame["sample_id"].tolist() != expected_test_ids:
        return False, None
    if not (frame["official_fold"].to_numpy() == fold).all():
        return False, None
    prediction = frame["clipped_prediction"].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        return False, None
    return True, prediction


def train_one_fold(
    fold: int,
    fold_info: dict[str, Any],
    cache_dir: Path,
    output_dir: Path,
    reference_feature_names: list[str],
    band_column: str,
) -> Any:
    start = time.perf_counter()
    print("\n" + "=" * 78)
    print(f"Training official fold {fold}")
    print("=" * 78)

    x_train_raw, train_audit = load_fold_features(
        cache_dir, fold, "train", fold_info["train_ids"]
    )
    x_test_raw, test_audit = load_fold_features(
        cache_dir, fold, "test", fold_info["test_ids"]
    )
    if list(x_train_raw.columns) != reference_feature_names:
        raise AssertionError(f"Fold {fold} train features changed after preflight.")
    if list(x_test_raw.columns) != reference_feature_names:
        raise AssertionError(f"Fold {fold} test features changed after preflight.")

    y_train = fold_info["train_y"]
    y_test = fold_info["test_y"]
    raw_band_train = x_train_raw[band_column].copy()
    raw_band_test = x_test_raw[band_column].copy()

    # Deliberately reproduce the uploaded scripts: means are fit on the full
    # official outer train-and-validation fold before the internal 80/20 split.
    column_means = x_train_raw.mean(numeric_only=True)
    x_train = x_train_raw.fillna(column_means).fillna(0)
    x_test = x_test_raw.fillna(column_means).fillna(0)

    if x_train.shape[1] != EXPECTED_TOTAL_FEATURES or x_test.shape[1] != EXPECTED_TOTAL_FEATURES:
        raise AssertionError(f"Fold {fold} did not retain exactly 133 features.")
    if not np.isfinite(x_train.to_numpy(dtype=float)).all():
        raise AssertionError(f"Fold {fold} imputed train features contain NaN/infinity.")
    if not np.isfinite(x_test.to_numpy(dtype=float)).all():
        raise AssertionError(f"Fold {fold} imputed test features contain NaN/infinity.")

    all_positions = np.arange(len(x_train))
    fit_positions, validation_positions = train_test_split(
        all_positions,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        shuffle=True,
    )
    x_fit = x_train.iloc[fit_positions]
    y_fit = y_train[fit_positions]
    x_validation = x_train.iloc[validation_positions]
    y_validation = y_train[validation_positions]

    roles = np.full(len(x_train), "", dtype=object)
    roles[fit_positions] = "xgboost_fit"
    roles[validation_positions] = "early_stopping_validation"
    if (roles == "").any():
        raise AssertionError(f"Fold {fold} internal split left unassigned rows.")
    split_frame = pd.DataFrame(
        {
            "sample_id": fold_info["train_ids"],
            "official_fold": fold,
            "outer_train_position": np.arange(len(x_train)),
            "internal_role": roles,
        }
    )
    atomic_write_csv(
        split_frame,
        output_dir / "folds" / f"fold_{fold}_internal_split.csv",
    )

    fit_missing_count, fit_missing_rate = count_missing(
        raw_band_train.iloc[fit_positions]
    )
    validation_missing_count, validation_missing_rate = count_missing(
        raw_band_train.iloc[validation_positions]
    )
    outer_missing_count, outer_missing_rate = count_missing(raw_band_train)
    test_missing_count, test_missing_rate = count_missing(raw_band_test)
    bandcenter_mean = float(column_means[band_column])

    print(
        f"Fold {fold} sizes: fit={len(x_fit)}, validation={len(x_validation)}, "
        f"official_test={len(x_test)}"
    )
    print(
        f"BandCenter missing: outer_train={outer_missing_count}/{len(raw_band_train)} "
        f"({outer_missing_rate:.6%}), fit={fit_missing_count}/{len(fit_positions)} "
        f"({fit_missing_rate:.6%}), validation={validation_missing_count}/"
        f"{len(validation_positions)} ({validation_missing_rate:.6%}), "
        f"official_test={test_missing_count}/{len(raw_band_test)} "
        f"({test_missing_rate:.6%})"
    )
    print(f"BandCenter outer-train imputation mean: {bandcenter_mean:.12g}")

    dfit = xgb.DMatrix(x_fit, label=y_fit)
    dvalidation = xgb.DMatrix(x_validation, label=y_validation)
    dtest = xgb.DMatrix(x_test)
    evaluations = {}

    booster = xgb.train(
        params=XGB_PARAMS,
        dtrain=dfit,
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[(dvalidation, "validation")],
        evals_result=evaluations,
        early_stopping_rounds=EARLY_STOPPING_PATIENCE,
        verbose_eval=50,
    )

    history = [float(value) for value in evaluations["validation"]["mae"]]
    if not history:
        raise AssertionError(f"Fold {fold} has an empty validation history.")
    best_iteration = int(booster.best_iteration)
    best_validation_mae = float(booster.best_score)
    final_validation_mae = float(history[-1])
    early_stopping_triggered = len(history) < MAX_BOOST_ROUNDS

    raw_prediction = np.asarray(
        predict_best_iteration(booster, dtest, best_iteration),
        dtype=float,
    )
    if raw_prediction.shape != y_test.shape:
        raise AssertionError(
            f"Fold {fold} prediction shape {raw_prediction.shape} does not match "
            f"test labels {y_test.shape}."
        )
    if not np.isfinite(raw_prediction).all():
        raise AssertionError(f"Fold {fold} raw predictions contain NaN/infinity.")
    clipped_prediction = np.maximum(raw_prediction, 0.0)

    clipped_metrics = calculate_metrics(y_test, clipped_prediction)
    raw_metrics = calculate_metrics(y_test, raw_prediction)
    elapsed_seconds = float(time.perf_counter() - start)

    metric_row = {
        "official_fold": fold,
        "n_outer_train": len(x_train),
        "n_xgboost_fit": len(x_fit),
        "n_internal_validation": len(x_validation),
        "n_official_test": len(x_test),
        "n_features": x_train.shape[1],
        "best_iteration_zero_based": best_iteration,
        "best_boosting_round_one_based": best_iteration + 1,
        "best_validation_mae": best_validation_mae,
        "final_validation_mae": final_validation_mae,
        "validation_rounds_evaluated": len(history),
        "early_stopping_triggered": early_stopping_triggered,
        "bandcenter_column_name": band_column,
        "bandcenter_missing_count_outer_train": outer_missing_count,
        "bandcenter_missing_rate_outer_train": outer_missing_rate,
        "bandcenter_missing_count_xgboost_fit": fit_missing_count,
        "bandcenter_missing_rate_xgboost_fit": fit_missing_rate,
        "bandcenter_missing_count_internal_validation": validation_missing_count,
        "bandcenter_missing_rate_internal_validation": validation_missing_rate,
        "bandcenter_missing_count_official_test": test_missing_count,
        "bandcenter_missing_rate_official_test": test_missing_rate,
        "bandcenter_imputation_mean_outer_train": bandcenter_mean,
        "mae": clipped_metrics["mae"],
        "rmse": clipped_metrics["rmse"],
        "r2": clipped_metrics["r2"],
        "raw_mae": raw_metrics["mae"],
        "raw_rmse": raw_metrics["rmse"],
        "raw_r2": raw_metrics["r2"],
        "elapsed_seconds": elapsed_seconds,
        "train_element_property_index_mode": train_audit[
            "element_property_index_mode"
        ],
        "train_bandcenter_index_mode": train_audit["bandcenter_index_mode"],
        "test_element_property_index_mode": test_audit[
            "element_property_index_mode"
        ],
        "test_bandcenter_index_mode": test_audit["bandcenter_index_mode"],
    }

    prediction_frame = pd.DataFrame(
        {
            "sample_id": fold_info["test_ids"],
            "official_fold": fold,
            "true_label": y_test,
            "raw_prediction": raw_prediction,
            "clipped_prediction": clipped_prediction,
            "best_iteration_zero_based": best_iteration,
            "best_validation_mae": best_validation_mae,
            "final_validation_mae": final_validation_mae,
            "early_stopping_triggered": early_stopping_triggered,
            "bandcenter_missing_rate_outer_train": outer_missing_rate,
            "bandcenter_missing_rate_official_test": test_missing_rate,
            "bandcenter_imputation_mean_outer_train": bandcenter_mean,
            "fold_mae": clipped_metrics["mae"],
            "fold_rmse": clipped_metrics["rmse"],
            "fold_r2": clipped_metrics["r2"],
        }
    )

    fold_dir = output_dir / "folds"
    atomic_write_csv(
        prediction_frame,
        fold_dir / f"fold_{fold}_predictions.csv",
    )
    atomic_write_json(metric_row, fold_dir / f"fold_{fold}_metrics.json")
    atomic_write_csv(
        pd.DataFrame(
            {
                "boosting_iteration_zero_based": np.arange(len(history)),
                "validation_mae": history,
            }
        ),
        fold_dir / f"fold_{fold}_validation_history.csv",
    )
    booster.save_model(str(fold_dir / f"fold_{fold}_model.json"))

    # Rebuild the combined files immediately so completed folds survive a later
    # interruption.
    rebuild_combined_outputs(output_dir)

    print(
        f"Fold {fold} complete: MAE={clipped_metrics['mae']:.6f} eV, "
        f"RMSE={clipped_metrics['rmse']:.6f} eV, R²={clipped_metrics['r2']:.6f}, "
        f"best_iteration={best_iteration}, early_stopping={early_stopping_triggered}"
    )
    print(
        f"Fold {fold} predictions written immediately to "
        f"{fold_dir / f'fold_{fold}_predictions.csv'}"
    )
    return clipped_prediction


def create_summary(output_dir: Path) -> tuple[Any, dict[str, float]]:
    metrics = pd.read_csv(output_dir / FOLD_METRICS_NAME)
    if metrics["official_fold"].tolist() != list(EXPECTED_FOLDS):
        raise AssertionError("Fold metric rows are incomplete or out of order.")

    rows = []
    summary_values = {}
    for metric in ("mae", "rmse", "r2"):
        values = metrics[metric].to_numpy(dtype=float)
        mean_value = float(np.mean(values))
        sd_value = float(np.std(values, ddof=1))
        rows.append(
            {
                "model": "Level 1 + BandCenter",
                "features": EXPECTED_TOTAL_FEATURES,
                "metric": metric.upper() if metric != "r2" else "R²",
                "fold_mean": mean_value,
                "fold_sd_ddof_1": sd_value,
                "formatted_mean_plus_minus_sd": f"{mean_value:.6f} ± {sd_value:.6f}",
            }
        )
        summary_values[f"{metric}_mean"] = mean_value
        summary_values[f"{metric}_fold_sd"] = sd_value

    summary = pd.DataFrame(rows)
    atomic_write_csv(summary, output_dir / SUMMARY_METRICS_NAME)
    return summary, summary_values


def make_audit(
    output_dir: Path,
    cache_dir: Path,
    config: dict[str, Any],
    cache_audits: list[dict[str, Any]],
    summary_values: dict[str, float],
) -> str:
    fold_metrics = pd.read_csv(output_dir / FOLD_METRICS_NAME)
    cache_audit_df = pd.DataFrame(cache_audits)

    fold_table_lines = [
        "| Fold | Features | BC missing outer train | BC missing official test | "
        "BC mean | Best iter (0-based) | MAE | RMSE | R² |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_metrics.itertuples(index=False):
        fold_table_lines.append(
            f"| {int(row.official_fold)} | {int(row.n_features)} | "
            f"{int(row.bandcenter_missing_count_outer_train)} "
            f"({row.bandcenter_missing_rate_outer_train:.6%}) | "
            f"{int(row.bandcenter_missing_count_official_test)} "
            f"({row.bandcenter_missing_rate_official_test:.6%}) | "
            f"{row.bandcenter_imputation_mean_outer_train:.10g} | "
            f"{int(row.best_iteration_zero_based)} | {row.mae:.6f} | "
            f"{row.rmse:.6f} | {row.r2:.6f} |"
        )

    cache_lines = [
        "| Fold | Split | Rows | BC missing | BC missing rate | Cache index mode |",
        "|---:|:---|---:|---:|---:|:---|",
    ]
    for row in cache_audit_df.itertuples(index=False):
        cache_lines.append(
            f"| {int(row.official_fold)} | {row.split} | {int(row.rows)} | "
            f"{int(row.bandcenter_missing_count)} | "
            f"{row.bandcenter_missing_rate:.6%} | "
            f"{row.bandcenter_index_mode} |"
        )

    text = f"""# Level 1 + BandCenter training audit

## Scope

This audit covers only the independent 133-feature XGBoost training run.
The Level-1 / Level-2 frozen-result comparison is intentionally deferred to a
separate comparison script.

## Frozen feature source

- Cache directory: `{cache_dir.resolve()}`
- Level-1 source: `fold_{{fold}}_{{train|test}}_ElementProperty.pkl`
- BandCenter source: `fold_{{fold}}_{{train|test}}_BandCenter.pkl`
- BandCenter was read directly from the same pre-imputation, per-featurizer
  cache convention used by the uploaded frozen Level-2 pipeline.
- No matminer featurizer was instantiated and no missing cache was regenerated.
- Exact file paths and SHA-256 hashes are stored in `{CACHE_MANIFEST_NAME}` and
  `{CONFIG_NAME}`.

## Feature assertions

- ElementProperty columns: **132**
- BandCenter columns: **1**
- Combined columns: **133**
- Structural cache files were never loaded.
- No features were sourced from: {", ".join(FORBIDDEN_FAMILY_NAMES)}.
- Train/test feature names and order were identical across all five folds.
- Feature provenance and order are stored in `{FEATURE_LIST_NAME}`.

## Sample and fold alignment

- Official MatBench folds were exactly `{list(EXPECTED_FOLDS)}`.
- Within every fold, official train/test sample IDs were unique and disjoint.
- The five official test folds covered the full sample universe exactly once.
- Cache row counts matched the corresponding ordered official split.
- Range-indexed frozen caches were bound positionally to the ordered sample IDs,
  matching the uploaded cache-generation pipeline.
- Official fold-assignment SHA-256:
  `{config["fold_assignment_sha256"]}`
- Assignments are stored in `{FOLD_ASSIGNMENTS_NAME}`.
- The realized internal 80/20 sample roles are stored in `{INTERNAL_SPLIT_NAME}`.

## Imputation policy

This run deliberately reproduced the uploaded baseline/ablation scripts:

1. Replace infinity with missing values.
2. Compute column means on the **full official outer train-and-validation fold**.
3. Fill the full outer train-and-validation fold and official test fold using
   those means.
4. Apply the uploaded script's final zero fallback for a still-all-missing column.
5. Only then split the imputed outer train-and-validation fold 80/20 with
   `random_state=42`.

The internal early-stopping validation subset therefore participates in the
outer-fold mean, exactly as in the uploaded scripts. No missingness indicator
was added.

## BandCenter missingness

{chr(10).join(cache_lines)}

## Fold results

All reported official-test metrics use non-negativity-clipped predictions.
Raw and clipped predictions are both stored.

{chr(10).join(fold_table_lines)}

## Five-fold summary

- MAE: **{summary_values["mae_mean"]:.6f} ± {summary_values["mae_fold_sd"]:.6f} eV**
- RMSE: **{summary_values["rmse_mean"]:.6f} ± {summary_values["rmse_fold_sd"]:.6f} eV**
- R²: **{summary_values["r2_mean"]:.6f} ± {summary_values["r2_fold_sd"]:.6f}**
- Fold SD uses `ddof=1`.

## Interpretation boundary

This training-only script does not determine how much of the frozen Level-1 to
Level-2 improvement is already present after adding BandCenter. That question,
including the paired fold differences and the “sequential gain under this
feature-addition order,” must be answered by the separate comparison script.
No result from this run should be interpreted as a strict causal contribution
of BandCenter.
"""
    atomic_write_text(text, output_dir / AUDIT_NAME)
    return text


def run(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    args.cache_dir = cache_dir
    args.output_dir = output_dir

    prepare_output_dir(output_dir, args.resume)
    log_path = output_dir / LOG_NAME
    log_mode = "a" if args.resume and log_path.exists() else "w"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open(log_mode, encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print("=" * 78)
            print("Level 1 + BandCenter XGBoost control")
            print("=" * 78)
            print(f"Started UTC: {utc_now()}")
            print(f"Cache directory: {cache_dir.resolve()}")
            print(f"Output directory: {output_dir.resolve()}")
            print(f"Resume mode: {args.resume}")
            print(
                "Imputation policy: full official outer-train-fold mean, then "
                "zero fallback, then internal 80/20 split."
            )

            load_runtime_dependencies()
            print("Runtime versions:")
            for key, value in collect_versions().items():
                print(f"  {key}: {value}")

            task = get_task()
            (
                fold_metadata,
                feature_names,
                band_column,
                assignments_df,
                fold_assignment_hash,
                cache_manifest,
                cache_audits,
            ) = preflight(task, cache_dir)

            feature_frame = feature_list_frame(feature_names, band_column)
            config = prepare_config(
                args,
                task,
                feature_names,
                band_column,
                fold_assignment_hash,
                cache_manifest,
            )
            validate_resume_config(output_dir, config, args.resume)

            atomic_write_csv(feature_frame, output_dir / FEATURE_LIST_NAME)
            atomic_write_csv(assignments_df, output_dir / FOLD_ASSIGNMENTS_NAME)
            atomic_write_csv(pd.DataFrame(cache_manifest), output_dir / CACHE_MANIFEST_NAME)
            atomic_write_json(config, output_dir / CONFIG_NAME)

            for fold in EXPECTED_FOLDS:
                if args.resume:
                    is_valid, saved_prediction = completed_fold_is_valid(
                        output_dir,
                        fold,
                        fold_metadata[fold]["test_ids"],
                    )
                    if is_valid:
                        print(f"Resume: fold {fold} is complete and validated; skipping.")
                        task.record(fold, saved_prediction)
                        continue

                clipped_prediction = train_one_fold(
                    fold,
                    fold_metadata[fold],
                    cache_dir,
                    output_dir,
                    feature_names,
                    band_column,
                )
                task.record(fold, clipped_prediction)

            rebuild_combined_outputs(output_dir)
            summary, summary_values = create_summary(output_dir)

            try:
                official_scores = task.scores
                atomic_write_json(official_scores, output_dir / MATBENCH_SCORES_NAME)
                print(f"MatBench recorded scores: {official_scores}")
            except Exception as exc:
                official_scores = {
                    "error": str(exc),
                    "note": (
                        "Locally calculated fold metrics remain available; "
                        "MatBench score serialization failed."
                    ),
                }
                atomic_write_json(official_scores, output_dir / MATBENCH_SCORES_NAME)
                print(f"Warning: MatBench task.scores failed: {exc}")

            make_audit(
                output_dir,
                cache_dir,
                config,
                cache_audits,
                summary_values,
            )

            print("\n" + "=" * 78)
            print("All five folds completed.")
            for row in summary.itertuples(index=False):
                print(f"{row.metric}: {row.formatted_mean_plus_minus_sd}")
            print(f"Combined predictions: {output_dir / PREDICTIONS_NAME}")
            print(f"Fold metrics: {output_dir / FOLD_METRICS_NAME}")
            print(f"Audit: {output_dir / AUDIT_NAME}")
            print(f"Finished UTC: {utc_now()}")
            print("=" * 78)
        except Exception:
            print("\nFATAL ERROR", file=sys.stderr)
            traceback.print_exc()
            print(
                "\nNo missing features were regenerated. Completed fold files, "
                "if any, remain available for --resume.",
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
