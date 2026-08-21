#!/usr/bin/env python3
"""
Error-tail and directional-confusion analysis for four frozen mp-gap models.

Models
------
1. Level 1
2. Level 1 + BandCenter
3. Level 2
4. Level 3

Input
-----
The script reads the fully aligned clipped-prediction table produced by
``compare_l1_bandcenter_l2_l3.py``:

    l1_bandcenter_l2_l3_aligned_predictions.csv

No model is trained, no prediction is changed, and no feature is generated.

Frozen manuscript definitions
-----------------------------
    true_zero = true_label == 0
    true_nonzero = true_label > 0

    predicted_near_zero = clipped_prediction < 0.1
    predicted_intermediate = (
        (clipped_prediction >= 0.1)
        & (clipped_prediction <= 0.5)
    )
    predicted_high = clipped_prediction > 0.5

    false_near_zero = true_nonzero & predicted_near_zero
    zero_gap_miss = true_zero & predicted_high

The historical machine-readable identifier ``false_near_zero`` is retained in
output filenames, columns, and configuration keys for compatibility. In human-
visible reporting it means a **positive-gap near-zero placement**; it does not
assert that every such placement is a large regression error.

The 0.1 and 0.5 eV boundaries are intentionally asymmetric. In particular,
``true_zero & prediction >= 0.1`` in the optional 2x2 table is NOT the
manuscript's ``zero_gap_miss``, which requires ``prediction > 0.5``.

Default paths
-------------
Input comparison directory:
    ../matbench_outputs_l1_bandcenter_l2_l3_comparison_run0731_ddof0

Historical comparison fallback:
    ../matbench_outputs_l1_bandcenter_l2_l3_comparison_run0731

New output directory:
    ../matbench_outputs_l1_bandcenter_l2_l3_error_analysis_run0731

Example
-------
    python analyze_l1_bandcenter_l2_l3_error_tails.py

Explicit paths:
    python analyze_l1_bandcenter_l2_l3_error_tails.py \
        --comparison-results-dir \
        ../matbench_outputs_l1_bandcenter_l2_l3_comparison_run0731_ddof0 \
        --output-dir \
        ../matbench_outputs_l1_bandcenter_l2_l3_error_analysis_run0731
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_NAME = "matbench_mp_gap"
EXPECTED_FOLDS = (0, 1, 2, 3, 4)

NEAR_ZERO_THRESHOLD_EV = 0.1
HIGH_THRESHOLD_EV = 0.5
TOP_FRACTIONS = (0.01, 0.05)

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

DEFAULT_COMPARISON_DIR = Path(
    "../matbench_outputs_l1_bandcenter_l2_l3_comparison_run0731_ddof0"
)
DEFAULT_COMPARISON_FALLBACK_DIR = Path(
    "../matbench_outputs_l1_bandcenter_l2_l3_comparison_run0731"
)
DEFAULT_OUTPUT_DIR = Path(
    "../matbench_outputs_l1_bandcenter_l2_l3_error_analysis_run0731"
)

ALIGNED_PREDICTIONS_NAME = "l1_bandcenter_l2_l3_aligned_predictions.csv"
COMPARISON_CONFIG_NAME = "l1_bandcenter_l2_l3_comparison_config.json"

SUMMARY_NAME = "l1_bandcenter_l2_l3_error_summary.csv"
QUANTILES_NAME = "l1_bandcenter_l2_l3_absolute_error_quantiles.csv"
TOP_SSE_NAME = "l1_bandcenter_l2_l3_top_sse_shares.csv"
TOP_SAMPLES_NAME = "l1_bandcenter_l2_l3_top5pct_samples.csv"
FALSE_NEAR_ZERO_NAME = "l1_bandcenter_l2_l3_false_near_zero_sse.csv"
DIRECTIONAL_ERRORS_NAME = "l1_bandcenter_l2_l3_directional_errors.csv"
CONFUSION_2X3_NAME = "l1_bandcenter_l2_l3_confusion_2x3_counts.csv"
CONFUSION_2X2_NAME = "l1_bandcenter_l2_l3_confusion_2x2_near_zero_counts.csv"
FALSE_NEAR_ZERO_SAMPLES_NAME = "l1_bandcenter_l2_l3_false_near_zero_samples.csv"
ZERO_GAP_MISS_SAMPLES_NAME = "l1_bandcenter_l2_l3_zero_gap_miss_samples.csv"
SOURCE_MANIFEST_NAME = "l1_bandcenter_l2_l3_error_analysis_source_manifest.csv"
CONFIG_NAME = "l1_bandcenter_l2_l3_error_analysis_config.json"
AUDIT_NAME = "l1_bandcenter_l2_l3_error_analysis_audit.md"
LOG_NAME = "l1_bandcenter_l2_l3_error_analysis_run.log"


# Loaded only after argparse has handled --help.
np = None
pd = None


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
            "Analyze absolute-error tails, SSE concentration, positive-gap "
            "near-zero placements, and exact 2x3 counts for the four frozen "
            "mp-gap models."
        )
    )
    parser.add_argument(
        "--comparison-results-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help=(
            "Directory containing the aligned four-model prediction CSV "
            f"(default: {DEFAULT_COMPARISON_DIR}; fallback: "
            f"{DEFAULT_COMPARISON_FALLBACK_DIR})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"New analysis output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    global np, pd
    try:
        import numpy as _np
        import pandas as _pd
    except ImportError as exc:
        raise RuntimeError(
            "Missing numpy/pandas. Activate the Python environment used for "
            f"the MatBench analysis. Original import error: {exc}"
        ) from exc
    np = _np
    pd = _pd


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
                f"Analysis output directory is not empty: {path}\n"
                "Choose a new --output-dir. Existing comparison/model results "
                "are never modified."
            )
    path.mkdir(parents=True, exist_ok=True)


def resolve_comparison_dir(requested: Path) -> Path:
    requested = requested.expanduser()
    requested_file = requested / ALIGNED_PREDICTIONS_NAME
    if requested_file.is_file():
        return requested

    use_default_fallback = requested == DEFAULT_COMPARISON_DIR
    fallback = DEFAULT_COMPARISON_FALLBACK_DIR.expanduser()
    fallback_file = fallback / ALIGNED_PREDICTIONS_NAME
    if use_default_fallback and fallback_file.is_file():
        print(
            f"Aligned predictions were not found under {requested}; using "
            f"historical comparison directory {fallback}. The prediction table "
            "is unchanged by the ddof convention."
        )
        return fallback

    checked = [str(requested_file)]
    if use_default_fallback:
        checked.append(str(fallback_file))
    formatted = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Aligned four-model prediction CSV was not found. Checked:\n"
        f"{formatted}"
    )


def load_comparison_config(comparison_dir: Path) -> dict[str, Any] | None:
    path = comparison_dir / COMPARISON_CONFIG_NAME
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("task_name") != TASK_NAME:
        raise AssertionError(
            f"Comparison config task is {config.get('task_name')!r}, expected "
            f"{TASK_NAME!r}."
        )
    return config


def validate_and_load_aligned(path: Path) -> Any:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    required = {
        "sample_id",
        "official_fold",
        "true_label",
        *(spec["prediction_column"] for spec in MODEL_SPECS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"Aligned prediction CSV is missing columns: {missing}")
    if frame.empty:
        raise AssertionError("Aligned prediction CSV is empty.")

    frame = frame.copy()
    frame["official_fold"] = pd.to_numeric(
        frame["official_fold"], errors="raise"
    ).astype(int)
    if tuple(sorted(frame["official_fold"].unique().tolist())) != EXPECTED_FOLDS:
        raise AssertionError(
            "Aligned prediction CSV does not contain exactly official folds "
            f"{EXPECTED_FOLDS}."
        )
    if frame["sample_id"].isna().any() or (frame["sample_id"] == "").any():
        raise AssertionError("Aligned prediction CSV contains missing sample IDs.")
    if frame["sample_id"].duplicated().any():
        raise AssertionError(
            "A sample ID appears more than once; expected each official test "
            "sample exactly once."
        )
    if frame.duplicated(["sample_id", "official_fold"]).any():
        raise AssertionError("Duplicate sample/fold rows were found.")

    numeric_columns = [
        "true_label",
        *(spec["prediction_column"] for spec in MODEL_SPECS),
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"Column {column!r} contains NaN/infinity.")
        if (values < 0).any():
            raise AssertionError(
                f"Column {column!r} contains negative values; frozen labels and "
                "clipped predictions must be nonnegative."
            )

    true_zero = frame["true_label"].to_numpy(dtype=float) == 0.0
    true_nonzero = frame["true_label"].to_numpy(dtype=float) > 0.0
    if not np.all(true_zero ^ true_nonzero):
        raise AssertionError(
            "true_zero and true_nonzero do not partition all rows exactly."
        )
    return frame


def fold_assignment_hash(frame) -> str:
    ordered = frame[["sample_id", "official_fold"]].sort_values(
        ["sample_id", "official_fold"], kind="stable"
    )
    canonical = "".join(
        f"{row.sample_id},{int(row.official_fold)}\n"
        for row in ordered.itertuples(index=False)
    )
    return sha256_text(canonical)


def safe_share(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def stable_error_order(model_frame):
    """Descending absolute error, then ascending sample ID, with stable sorting."""
    return model_frame.sort_values(
        ["absolute_error", "sample_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_model_frame(aligned, spec: dict[str, Any]) -> Any:
    frame = aligned[
        ["sample_id", "official_fold", "true_label", spec["prediction_column"]]
    ].copy()
    frame = frame.rename(columns={spec["prediction_column"]: "clipped_prediction"})
    frame["model"] = spec["label"]
    frame["features"] = spec["features"]
    frame["signed_error"] = (
        frame["clipped_prediction"] - frame["true_label"]
    )
    frame["absolute_error"] = frame["signed_error"].abs()
    frame["squared_error"] = frame["signed_error"] ** 2

    frame["true_zero"] = frame["true_label"] == 0.0
    frame["true_nonzero"] = frame["true_label"] > 0.0
    frame["predicted_near_zero"] = (
        frame["clipped_prediction"] < NEAR_ZERO_THRESHOLD_EV
    )
    frame["predicted_intermediate"] = (
        (frame["clipped_prediction"] >= NEAR_ZERO_THRESHOLD_EV)
        & (frame["clipped_prediction"] <= HIGH_THRESHOLD_EV)
    )
    frame["predicted_high"] = (
        frame["clipped_prediction"] > HIGH_THRESHOLD_EV
    )
    frame["false_near_zero"] = (
        frame["true_nonzero"] & frame["predicted_near_zero"]
    )
    frame["zero_gap_miss"] = frame["true_zero"] & frame["predicted_high"]

    predicted_partition_count = (
        frame[
            [
                "predicted_near_zero",
                "predicted_intermediate",
                "predicted_high",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )
    if not (predicted_partition_count == 1).all():
        raise AssertionError(
            f"{spec['label']} prediction bins do not partition every row exactly."
        )
    if not (frame[["true_zero", "true_nonzero"]].astype(int).sum(axis=1) == 1).all():
        raise AssertionError(
            f"{spec['label']} true-label classes do not partition every row exactly."
        )
    return frame


def calculate_quantiles(model_frame, spec: dict[str, Any]) -> dict[str, Any]:
    absolute = model_frame["absolute_error"].to_numpy(dtype=float)
    p50, p90, p99 = np.quantile(absolute, [0.50, 0.90, 0.99])
    maximum = float(np.max(absolute))
    maximum_mask = absolute == maximum
    maximum_ids = sorted(model_frame.loc[maximum_mask, "sample_id"].tolist())
    return {
        "model": spec["label"],
        "features": spec["features"],
        "n_samples": len(model_frame),
        "absolute_error_p50": float(p50),
        "absolute_error_p90": float(p90),
        "absolute_error_p99": float(p99),
        "absolute_error_max": maximum,
        "max_tie_count": int(maximum_mask.sum()),
        "max_sample_ids": "|".join(maximum_ids),
        "quantile_method": "numpy.quantile default linear interpolation",
    }


def calculate_top_sse(
    model_frame,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], Any]:
    ordered = stable_error_order(model_frame)
    ordered["absolute_error_rank_1_based"] = np.arange(1, len(ordered) + 1)
    n_total = len(ordered)
    total_sse = float(ordered["squared_error"].sum())
    rows = []
    selected_top5 = None

    for fraction in TOP_FRACTIONS:
        count = int(math.ceil(n_total * fraction))
        selected = ordered.iloc[:count].copy()
        selected_sse = float(selected["squared_error"].sum())
        rows.append(
            {
                "model": spec["label"],
                "features": spec["features"],
                "top_fraction": fraction,
                "top_percent": fraction * 100,
                "n_total": n_total,
                "n_selected_ceil": count,
                "selection_rule": (
                    "absolute_error descending; sample_id ascending for boundary ties"
                ),
                "selected_sse": selected_sse,
                "total_sse": total_sse,
                "selected_sse_share": safe_share(selected_sse, total_sse),
            }
        )
        if fraction == 0.05:
            selected_top5 = selected

    if selected_top5 is None:
        raise AssertionError("Top-5% selection was not constructed.")
    top1_count = int(math.ceil(n_total * 0.01))
    selected_top5 = selected_top5.copy()
    selected_top5["in_top_1_percent"] = (
        selected_top5["absolute_error_rank_1_based"] <= top1_count
    )
    selected_top5["selection_scope"] = "top_5_percent_by_model"
    return rows, selected_top5


def calculate_confusion_2x3(model_frame, spec: dict[str, Any]) -> Any:
    true_classes = (
        ("true zero (y == 0)", model_frame["true_zero"]),
        ("true nonzero (y > 0)", model_frame["true_nonzero"]),
    )
    prediction_classes = (
        ("pred_lt_0_1_count", model_frame["predicted_near_zero"]),
        ("pred_0_1_to_0_5_count", model_frame["predicted_intermediate"]),
        ("pred_gt_0_5_count", model_frame["predicted_high"]),
    )
    rows = []
    for true_label, true_mask in true_classes:
        row = {
            "model": spec["label"],
            "features": spec["features"],
            "true_label_class": true_label,
        }
        for column, predicted_mask in prediction_classes:
            row[column] = int((true_mask & predicted_mask).sum())
        row["row_total"] = int(true_mask.sum())
        if (
            row["pred_lt_0_1_count"]
            + row["pred_0_1_to_0_5_count"]
            + row["pred_gt_0_5_count"]
            != row["row_total"]
        ):
            raise AssertionError(
                f"{spec['label']} 2x3 row counts do not sum to the row total."
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    if int(result["row_total"].sum()) != len(model_frame):
        raise AssertionError(f"{spec['label']} 2x3 counts do not sum to N.")
    return result


def calculate_confusion_2x2(model_frame, spec: dict[str, Any]) -> Any:
    true_classes = (
        ("true zero (y == 0)", model_frame["true_zero"]),
        ("true nonzero (y > 0)", model_frame["true_nonzero"]),
    )
    predicted_near = model_frame["predicted_near_zero"]
    predicted_not_near = ~predicted_near
    rows = []
    for true_label, true_mask in true_classes:
        near_count = int((true_mask & predicted_near).sum())
        not_near_count = int((true_mask & predicted_not_near).sum())
        rows.append(
            {
                "model": spec["label"],
                "features": spec["features"],
                "true_label_class": true_label,
                "pred_lt_0_1_count": near_count,
                "pred_ge_0_1_count": not_near_count,
                "row_total": int(true_mask.sum()),
                "warning": (
                    "true zero & pred >= 0.1 is not zero_gap_miss; "
                    "zero_gap_miss requires pred > 0.5"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if int(result["row_total"].sum()) != len(model_frame):
        raise AssertionError(f"{spec['label']} 2x2 counts do not sum to N.")
    return result


def calculate_directional_errors(
    model_frame,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    n_total = len(model_frame)
    n_true_zero = int(model_frame["true_zero"].sum())
    n_true_nonzero = int(model_frame["true_nonzero"].sum())
    false_mask = model_frame["false_near_zero"]
    miss_mask = model_frame["zero_gap_miss"]

    false_count = int(false_mask.sum())
    miss_count = int(miss_mask.sum())
    combined_count = false_count + miss_count
    if (false_mask & miss_mask).any():
        raise AssertionError(
            f"{spec['label']} positive-gap near-zero and zero-gap-miss masks "
            "overlap."
        )

    total_sse = float(model_frame["squared_error"].sum())
    false_sse = float(model_frame.loc[false_mask, "squared_error"].sum())
    miss_sse = float(model_frame.loc[miss_mask, "squared_error"].sum())
    combined_sse = false_sse + miss_sse

    false_true_values = model_frame.loc[false_mask, "true_label"].to_numpy(
        dtype=float
    )
    false_true_median = (
        float(np.median(false_true_values))
        if len(false_true_values)
        else float("nan")
    )

    false_row = {
        "model": spec["label"],
        "features": spec["features"],
        "n_total": n_total,
        "n_true_nonzero": n_true_nonzero,
        "false_near_zero_count": false_count,
        "false_near_zero_rate_among_true_nonzero": safe_share(
            false_count, n_true_nonzero
        ),
        "false_near_zero_rate_total": safe_share(false_count, n_total),
        "false_near_zero_sse": false_sse,
        "total_sse": total_sse,
        "false_near_zero_sse_share": safe_share(false_sse, total_sse),
        "false_near_zero_true_label_median_ev": false_true_median,
        "definition": "true_label > 0 and clipped_prediction < 0.1",
    }

    directional_row = {
        "model": spec["label"],
        "features": spec["features"],
        "n_total": n_total,
        "n_true_zero": n_true_zero,
        "n_true_nonzero": n_true_nonzero,
        "zero_gap_miss_count": miss_count,
        "zero_gap_miss_rate_among_true_zero": safe_share(
            miss_count, n_true_zero
        ),
        "zero_gap_miss_rate_total": safe_share(miss_count, n_total),
        "zero_gap_miss_sse": miss_sse,
        "zero_gap_miss_sse_share": safe_share(miss_sse, total_sse),
        "false_near_zero_count": false_count,
        "false_near_zero_rate_among_true_nonzero": safe_share(
            false_count, n_true_nonzero
        ),
        "false_near_zero_rate_total": safe_share(false_count, n_total),
        "false_near_zero_sse": false_sse,
        "false_near_zero_sse_share": safe_share(false_sse, total_sse),
        "combined_directional_error_count": combined_count,
        "combined_directional_error_rate_total": safe_share(
            combined_count, n_total
        ),
        "combined_directional_error_sse": combined_sse,
        "combined_directional_error_sse_share": safe_share(
            combined_sse, total_sse
        ),
        "total_sse": total_sse,
    }

    sample_columns = [
        "model",
        "features",
        "sample_id",
        "official_fold",
        "true_label",
        "clipped_prediction",
        "signed_error",
        "absolute_error",
        "squared_error",
    ]
    false_samples = model_frame.loc[false_mask, sample_columns].copy()
    false_samples = false_samples.sort_values(
        ["absolute_error", "sample_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    miss_samples = model_frame.loc[miss_mask, sample_columns].copy()
    miss_samples = miss_samples.sort_values(
        ["absolute_error", "sample_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    return false_row, directional_row, false_samples, miss_samples


def build_summary(
    quantiles,
    top_sse,
    false_near_zero,
    directional_errors,
) -> Any:
    q = quantiles.set_index("model")
    top = top_sse.pivot(index="model", columns="top_fraction")
    false = false_near_zero.set_index("model")
    directional = directional_errors.set_index("model")
    rows = []
    for spec in MODEL_SPECS:
        model = spec["label"]
        rows.append(
            {
                "model": model,
                "features": spec["features"],
                "n_samples": int(q.loc[model, "n_samples"]),
                "absolute_error_p50": q.loc[model, "absolute_error_p50"],
                "absolute_error_p90": q.loc[model, "absolute_error_p90"],
                "absolute_error_p99": q.loc[model, "absolute_error_p99"],
                "absolute_error_max": q.loc[model, "absolute_error_max"],
                "top_1_percent_n": int(
                    top.loc[model, ("n_selected_ceil", 0.01)]
                ),
                "top_1_percent_sse_share": top.loc[
                    model, ("selected_sse_share", 0.01)
                ],
                "top_5_percent_n": int(
                    top.loc[model, ("n_selected_ceil", 0.05)]
                ),
                "top_5_percent_sse_share": top.loc[
                    model, ("selected_sse_share", 0.05)
                ],
                "false_near_zero_count": int(
                    false.loc[model, "false_near_zero_count"]
                ),
                "false_near_zero_sse_share": false.loc[
                    model, "false_near_zero_sse_share"
                ],
                "false_near_zero_true_label_median_ev": false.loc[
                    model, "false_near_zero_true_label_median_ev"
                ],
                "zero_gap_miss_count": int(
                    directional.loc[model, "zero_gap_miss_count"]
                ),
                "combined_directional_error_rate_total": directional.loc[
                    model, "combined_directional_error_rate_total"
                ],
            }
        )
    return pd.DataFrame(rows)


def markdown_2x3_table(confusion, model: str) -> str:
    part = confusion.loc[confusion["model"] == model]
    lines = [
        f"### {model}",
        "",
        "| True-label class | pred < 0.1 | 0.1 ≤ pred ≤ 0.5 | pred > 0.5 | Row total |",
        "|:---|---:|---:|---:|---:|",
    ]
    for row in part.itertuples(index=False):
        lines.append(
            f"| {row.true_label_class} | {int(row.pred_lt_0_1_count)} | "
            f"{int(row.pred_0_1_to_0_5_count)} | "
            f"{int(row.pred_gt_0_5_count)} | {int(row.row_total)} |"
        )
    return "\n".join(lines)


def markdown_summary_table(summary) -> str:
    lines = [
        "| Model | P50 | P90 | P99 | Max | Top 1% SSE | Top 5% SSE | "
        "Positive-gap near-zero SSE |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.absolute_error_p50:.6f} | "
            f"{row.absolute_error_p90:.6f} | {row.absolute_error_p99:.6f} | "
            f"{row.absolute_error_max:.6f} | "
            f"{row.top_1_percent_sse_share:.6%} | "
            f"{row.top_5_percent_sse_share:.6%} | "
            f"{row.false_near_zero_sse_share:.6%} |"
        )
    return "\n".join(lines)


def markdown_directional_table(directional) -> str:
    lines = [
        "| Model | Zero-gap miss count / true-zero | Miss / N | "
        "Positive-gap near-zero count / true-positive-gap | "
        "Positive-gap near-zero / N | Combined / N |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for row in directional.itertuples(index=False):
        lines.append(
            f"| {row.model} | {int(row.zero_gap_miss_count)} / "
            f"{int(row.n_true_zero)} "
            f"({row.zero_gap_miss_rate_among_true_zero:.6%}) | "
            f"{row.zero_gap_miss_rate_total:.6%} | "
            f"{int(row.false_near_zero_count)} / {int(row.n_true_nonzero)} "
            f"({row.false_near_zero_rate_among_true_nonzero:.6%}) | "
            f"{row.false_near_zero_rate_total:.6%} | "
            f"{row.combined_directional_error_rate_total:.6%} |"
        )
    return "\n".join(lines)


def make_audit(
    output_dir: Path,
    comparison_dir: Path,
    aligned_path: Path,
    comparison_config: dict[str, Any] | None,
    assignment_hash: str,
    summary,
    top_sse,
    false_near_zero,
    directional,
    confusion_2x3,
) -> None:
    config_note = (
        "The comparison config was loaded and its task/fold hash was verified."
        if comparison_config is not None
        else (
            "No comparison config was present; the aligned table itself was "
            "strictly validated."
        )
    )
    confusion_sections = "\n\n".join(
        markdown_2x3_table(confusion_2x3, spec["label"])
        for spec in MODEL_SPECS
    )

    false_lines = [
        "| Model | Count | True-label median | Positive-gap near-zero SSE | "
        "Total SSE | Positive-gap near-zero SSE share |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for row in false_near_zero.itertuples(index=False):
        false_lines.append(
            f"| {row.model} | {int(row.false_near_zero_count)} | "
            f"{row.false_near_zero_true_label_median_ev:.6f} | "
            f"{row.false_near_zero_sse:.6f} | {row.total_sse:.6f} | "
            f"{row.false_near_zero_sse_share:.6%} |"
        )

    top_lines = [
        "| Model | Top fraction | Selected N | Selected SSE | Total SSE | SSE share |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for row in top_sse.itertuples(index=False):
        top_lines.append(
            f"| {row.model} | {row.top_percent:.0f}% | "
            f"{int(row.n_selected_ceil)} | {row.selected_sse:.6f} | "
            f"{row.total_sse:.6f} | {row.selected_sse_share:.6%} |"
        )

    text = f"""# Four-model error-tail analysis audit

## Source and alignment

- Comparison directory: `{comparison_dir.resolve()}`
- Aligned clipped predictions: `{aligned_path.resolve()}`
- Input SHA-256: `{sha256_file(aligned_path)}`
- Official fold-assignment SHA-256: `{assignment_hash}`
- {config_note}
- Every sample ID was unique and appeared exactly once.
- Official folds were exactly `{list(EXPECTED_FOLDS)}`.
- True labels and all four frozen clipped predictions were finite and nonnegative.
- No model was trained and no prediction was modified.

## Frozen manuscript definitions

```python
true_zero = true_label == 0
true_nonzero = true_label > 0

predicted_near_zero = clipped_prediction < 0.1
predicted_intermediate = (
    (clipped_prediction >= 0.1)
    & (clipped_prediction <= 0.5)
)
predicted_high = clipped_prediction > 0.5

false_near_zero = true_nonzero & predicted_near_zero
zero_gap_miss = true_zero & predicted_high
```

For backward compatibility, machine-readable filenames, columns, and config
keys retain the historical identifier `false_near_zero`. Human-visible text
calls this event a **positive-gap near-zero placement**; the identifier does not
assert that every event is a large regression error.

No `true_label >= 0.5` condition was used. The 0.1 and 0.5 eV boundaries
remain asymmetric and the intermediate dead band is retained.

## Absolute-error tails and SSE concentration

- P50/P90/P99/max use each model's full-sample absolute errors.
- Quantiles use NumPy's default linear interpolation.
- Top 1% and 5% are selected separately for each model.
- Selected counts are `ceil(N × fraction)`.
- Ordering is absolute error descending, then sample ID ascending; stable
  mergesort makes boundary ties reproducible.
- SSE share is selected-subset `sum(error²)` divided by full-model `sum(error²)`.

{markdown_summary_table(summary)}

{chr(10).join(top_lines)}

## Main 2×3 exact count tables

The upper-right cell is `zero_gap_miss`. The lower-left cell is the positive-gap
near-zero placement event stored under the compatibility field
`false_near_zero`.

{confusion_sections}

## Directional error rates

{markdown_directional_table(directional)}

## Positive-gap near-zero placement SSE

{chr(10).join(false_lines)}

## Optional 2×2 table warning

`{CONFUSION_2X2_NAME}` uses the single prediction split `pred < 0.1` versus
`pred >= 0.1`. Its `true zero & pred >= 0.1` cell is **not** the manuscript's
`zero_gap_miss`; the manuscript miss requires `true_label == 0` and
`prediction > 0.5`.

## Output inventory

- Overall summary: `{SUMMARY_NAME}`
- Quantiles: `{QUANTILES_NAME}`
- Top SSE shares: `{TOP_SSE_NAME}`
- Selected top-5% sample rows: `{TOP_SAMPLES_NAME}`
- Positive-gap near-zero SSE (historical filename): `{FALSE_NEAR_ZERO_NAME}`
- Directional rates/counts: `{DIRECTIONAL_ERRORS_NAME}`
- Main 2×3 counts: `{CONFUSION_2X3_NAME}`
- Optional 2×2 counts: `{CONFUSION_2X2_NAME}`
- Positive-gap near-zero samples (historical filename): `{FALSE_NEAR_ZERO_SAMPLES_NAME}`
- Zero-gap-miss samples: `{ZERO_GAP_MISS_SAMPLES_NAME}`
"""
    atomic_write_text(text, output_dir / AUDIT_NAME)


def run(args: argparse.Namespace) -> None:
    requested_comparison_dir = args.comparison_results_dir.expanduser()
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
            print("Four-model frozen clipped-prediction error-tail analysis")
            print("=" * 78)
            print(f"Started UTC: {utc_now()}")
            print(f"Output directory: {output_dir.resolve()}")

            load_runtime_dependencies()
            versions = collect_versions()
            for name, version in versions.items():
                print(f"  {name}: {version}")

            comparison_dir = resolve_comparison_dir(requested_comparison_dir)
            aligned_path = comparison_dir / ALIGNED_PREDICTIONS_NAME
            comparison_config = load_comparison_config(comparison_dir)
            aligned = validate_and_load_aligned(aligned_path)
            assignment_hash = fold_assignment_hash(aligned)

            if comparison_config is not None:
                saved_hash = comparison_config.get("fold_assignment_sha256")
                if saved_hash != assignment_hash:
                    raise AssertionError(
                        "Aligned prediction fold hash does not match the comparison config."
                    )

            print(f"Resolved comparison directory: {comparison_dir.resolve()}")
            print(f"Aligned samples: {len(aligned):,}")
            print(f"Official fold hash: {assignment_hash}")

            quantile_rows = []
            top_sse_rows = []
            top_sample_frames = []
            false_rows = []
            directional_rows = []
            confusion_2x3_frames = []
            confusion_2x2_frames = []
            false_sample_frames = []
            miss_sample_frames = []

            for spec in MODEL_SPECS:
                model_frame = build_model_frame(aligned, spec)
                quantile_rows.append(calculate_quantiles(model_frame, spec))

                model_top_rows, top_samples = calculate_top_sse(model_frame, spec)
                top_sse_rows.extend(model_top_rows)
                top_sample_frames.append(top_samples)

                confusion_2x3_frames.append(
                    calculate_confusion_2x3(model_frame, spec)
                )
                confusion_2x2_frames.append(
                    calculate_confusion_2x2(model_frame, spec)
                )

                (
                    false_row,
                    directional_row,
                    false_samples,
                    miss_samples,
                ) = calculate_directional_errors(model_frame, spec)
                false_rows.append(false_row)
                directional_rows.append(directional_row)
                false_sample_frames.append(false_samples)
                miss_sample_frames.append(miss_samples)

            quantiles = pd.DataFrame(quantile_rows)
            top_sse = pd.DataFrame(top_sse_rows)
            top_samples = pd.concat(top_sample_frames, ignore_index=True)
            false_near_zero = pd.DataFrame(false_rows)
            directional = pd.DataFrame(directional_rows)
            confusion_2x3 = pd.concat(confusion_2x3_frames, ignore_index=True)
            confusion_2x2 = pd.concat(confusion_2x2_frames, ignore_index=True)
            false_samples = pd.concat(false_sample_frames, ignore_index=True)
            miss_samples = pd.concat(miss_sample_frames, ignore_index=True)
            summary = build_summary(
                quantiles, top_sse, false_near_zero, directional
            )

            atomic_write_csv(summary, output_dir / SUMMARY_NAME)
            atomic_write_csv(quantiles, output_dir / QUANTILES_NAME)
            atomic_write_csv(top_sse, output_dir / TOP_SSE_NAME)
            atomic_write_csv(top_samples, output_dir / TOP_SAMPLES_NAME)
            atomic_write_csv(false_near_zero, output_dir / FALSE_NEAR_ZERO_NAME)
            atomic_write_csv(directional, output_dir / DIRECTIONAL_ERRORS_NAME)
            atomic_write_csv(confusion_2x3, output_dir / CONFUSION_2X3_NAME)
            atomic_write_csv(confusion_2x2, output_dir / CONFUSION_2X2_NAME)
            atomic_write_csv(
                false_samples, output_dir / FALSE_NEAR_ZERO_SAMPLES_NAME
            )
            atomic_write_csv(
                miss_samples, output_dir / ZERO_GAP_MISS_SAMPLES_NAME
            )

            manifest_rows = [
                {
                    "source_type": "aligned_frozen_clipped_predictions",
                    "path": str(aligned_path.resolve()),
                    "size_bytes": aligned_path.stat().st_size,
                    "sha256": sha256_file(aligned_path),
                }
            ]
            comparison_config_path = comparison_dir / COMPARISON_CONFIG_NAME
            if comparison_config_path.is_file():
                manifest_rows.append(
                    {
                        "source_type": "comparison_config",
                        "path": str(comparison_config_path.resolve()),
                        "size_bytes": comparison_config_path.stat().st_size,
                        "sha256": sha256_file(comparison_config_path),
                    }
                )
            manifest = pd.DataFrame(manifest_rows)
            atomic_write_csv(manifest, output_dir / SOURCE_MANIFEST_NAME)

            config = {
                "analysis_name": "four-model error tails and directional confusion",
                "task_name": TASK_NAME,
                "created_utc": utc_now(),
                "requested_comparison_dir": str(requested_comparison_dir),
                "resolved_comparison_dir": str(comparison_dir.resolve()),
                "aligned_predictions_path": str(aligned_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "models": list(MODEL_SPECS),
                "definitions": {
                    "true_zero": "true_label == 0",
                    "true_nonzero": "true_label > 0",
                    "predicted_near_zero": "clipped_prediction < 0.1",
                    "predicted_intermediate": (
                        "clipped_prediction >= 0.1 and clipped_prediction <= 0.5"
                    ),
                    "predicted_high": "clipped_prediction > 0.5",
                    "false_near_zero": (
                        "true_label > 0 and clipped_prediction < 0.1"
                    ),
                    "zero_gap_miss": (
                        "true_label == 0 and clipped_prediction > 0.5"
                    ),
                },
                "thresholds_ev": {
                    "near_zero_strict_upper": NEAR_ZERO_THRESHOLD_EV,
                    "intermediate_inclusive_lower": NEAR_ZERO_THRESHOLD_EV,
                    "intermediate_inclusive_upper": HIGH_THRESHOLD_EV,
                    "high_strict_lower": HIGH_THRESHOLD_EV,
                },
                "tail_statistics": {
                    "absolute_error_quantiles": [0.50, 0.90, 0.99],
                    "quantile_method": "numpy.quantile default linear interpolation",
                    "top_fractions": list(TOP_FRACTIONS),
                    "selected_n": "ceil(N * fraction)",
                    "ranking": (
                        "absolute_error descending, then sample_id ascending, "
                        "stable mergesort"
                    ),
                    "sse_share": "selected sum(error^2) / full-model sum(error^2)",
                },
                "fold_assignment_sha256": assignment_hash,
                "input_sha256": sha256_file(aligned_path),
                "versions": versions,
            }
            atomic_write_json(config, output_dir / CONFIG_NAME)
            make_audit(
                output_dir,
                comparison_dir,
                aligned_path,
                comparison_config,
                assignment_hash,
                summary,
                top_sse,
                false_near_zero,
                directional,
                confusion_2x3,
            )

            print("\n" + markdown_summary_table(summary))
            print("\n" + markdown_directional_table(directional))
            for spec in MODEL_SPECS:
                print("\n" + markdown_2x3_table(confusion_2x3, spec["label"]))
            print(f"\nSummary: {output_dir / SUMMARY_NAME}")
            print(f"Main 2x3 counts: {output_dir / CONFUSION_2X3_NAME}")
            print(f"Audit: {output_dir / AUDIT_NAME}")
            print(f"Finished UTC: {utc_now()}")
            print("=" * 78)
        except Exception:
            print("\nFATAL ERROR", file=sys.stderr)
            traceback.print_exc()
            print(
                "\nNo source comparison/model result was modified.",
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
