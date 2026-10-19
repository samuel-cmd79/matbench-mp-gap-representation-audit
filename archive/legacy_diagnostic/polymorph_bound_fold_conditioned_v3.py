#!/usr/bin/env python3
"""
Fold-conditioned oracle bounds for repeated-composition entries.

This script is a non-destructive successor to ``polymorph_analysis_v3.py``.
It does not modify the old script or write to the old ``polymorph_analysis``
directory.

The script reports two different oracle quantities:

1. Global fixed-function oracle bound

       B_global = (1 / N) * sum_c sum_{i in G_c} |y_i - median(y_{G_c})|

   This is the MAE optimum for one fixed composition function f(c) across the
   complete pooled dataset.  It is retained only to reproduce the old result.

2. Fold-conditioned OOF oracle bound

       B_fold = (1 / N) * sum_{c,k} sum_{i in G_{c,k}}
                |y_i - median(y_{G_{c,k}})|

   This is the strict oracle bound that matches pooled official MatBench OOF
   predictions, because each outer fold may use a different function f_k(c).

Two populations are kept separate throughout:

* global_repeated_formula_subset:
  entries whose reduced formula occurs at least twice in the complete dataset.
* same_fold_repeated_subset:
  entries whose (reduced_formula, official_test_fold) group has size >= 2.

L1/L2 predictions are legacy ID-less ``.npy`` arrays.  Their IDs are rebuilt
from the official MatBench test-fold order, as justified by the reviewed
generation scripts, then every subsequent join is performed by stable ``mbid``.
L3 predictions are joined directly using IDs embedded in each ``.npz``.

Default outputs (all in a new directory):

* polymorph_bound_prediction_manifest.csv
* polymorph_bound_summary.csv
* polymorph_bound_by_spread.csv
* polymorph_bound_group_size_distribution.csv
* polymorph_bound_bootstrap.csv
* polymorph_bound_bootstrap_by_spread.csv
* polymorph_bound_l1_constancy_exceptions.csv (only if nonzero ranges exist)
* repeated_composition_groups_top.csv
* polymorph_bound_by_spread.png
* polymorph_bound_delta_ci_by_spread.png
* polymorph_bound_audit.md

Usage:

    python polymorph_bound_fold_conditioned.py

Optional quick validation without MatBench or prediction files:

    python polymorph_bound_fold_conditioned.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# User-editable defaults
# ============================================================================

TASK_NAME = "matbench_mp_gap"
MODEL = "xgb"
FOLDS = (0, 1, 2, 3, 4)

# Paths are resolved relative to this script by default.  CLI arguments can
# override all paths without editing the file.
V1_PRED_PATTERN = "../outputs_v1_run0709/predictions_{model}/pred_fold_{fold}.npy"
V2_PRED_PATTERN = "../matbench_outputs_v2_run0709/predictions_{model}/pred_fold_{fold}.npy"
V3_PRED_PATTERN = "../results_v4/fold_{fold}/test_preds_clipped.npz"
V3_PRED_KEY = "preds"
V3_ID_KEY = "ids"

# Deliberately different from the old ``./polymorph_analysis`` output path.
OUTPUT_DIR_NAME = "polymorph_bound_fold_conditioned_outputs"

SPREAD_EDGES = (0.0, 0.1, 0.5, 1.0, np.inf)
SPREAD_LABELS = ("<0.1 eV", "0.1–0.5 eV", "0.5–1 eV", ">1 eV")

L1_CONSTANCY_TOLERANCE_EV = 1e-9
COMPARISON_TOLERANCE_EV = 1e-12
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260730

# These are reference checks supplied by the existing analysis, not analysis
# results.  All reported values are recomputed from the loaded data.  The new
# fold-conditioned analysis is blocked unless this compatibility check passes.
LEGACY_REFERENCE_N_SAMPLES = 39737
LEGACY_REFERENCE_N_FORMULAS = 11788
LEGACY_REFERENCE_BOUND_DISPLAY = "0.1539"

POPULATION_A = "global_repeated_formula_subset"
POPULATION_B = "same_fold_repeated_subset"

# Acceptance references supplied for the frozen predictions. These values are
# validation targets only; reported estimates are always recomputed from data.
SAME_FOLD_SPREAD_DELTA_REFERENCE = {
    "<0.1 eV": {"L2_MAE_minus_bound_eV": 0.157999, "L3_MAE_minus_bound_eV": 0.081953},
    "0.1–0.5 eV": {"L2_MAE_minus_bound_eV": 0.198662, "L3_MAE_minus_bound_eV": 0.093959},
    "0.5–1 eV": {"L2_MAE_minus_bound_eV": 0.097801, "L3_MAE_minus_bound_eV": 0.000661},
    ">1 eV": {"L2_MAE_minus_bound_eV": 0.016485, "L3_MAE_minus_bound_eV": -0.082437},
}


@dataclass
class Config:
    project_dir: Path
    output_dir: Path
    task_name: str
    model: str
    folds: Tuple[int, ...]
    v1_pattern: str
    v2_pattern: str
    v3_pattern: str
    v3_pred_key: str
    v3_id_key: str
    l1_tolerance: float
    bootstrap_replicates: int
    bootstrap_seed: int
    make_plot: bool


@dataclass
class AssertionResult:
    name: str
    passed: bool
    detail: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Recompute global and fold-conditioned repeated-composition MAE oracle bounds."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=script_dir,
        help="Base directory used to resolve relative prediction paths (default: script directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "New output directory. Relative paths are resolved under --project-dir "
            f"(default: {OUTPUT_DIR_NAME})."
        ),
    )
    parser.add_argument("--task-name", default=TASK_NAME)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--v1-pattern", default=V1_PRED_PATTERN)
    parser.add_argument("--v2-pattern", default=V2_PRED_PATTERN)
    parser.add_argument("--v3-pattern", default=V3_PRED_PATTERN)
    parser.add_argument("--v3-pred-key", default=V3_PRED_KEY)
    parser.add_argument("--v3-id-key", default=V3_ID_KEY)
    parser.add_argument("--l1-tolerance", type=float, default=L1_CONSTANCY_TOLERANCE_EV)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    project_dir = args.project_dir.expanduser().resolve()
    if args.output_dir is None:
        output_dir = project_dir / OUTPUT_DIR_NAME
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = project_dir / output_dir
        output_dir = output_dir.resolve()

    n_bootstrap = 0 if args.skip_bootstrap else int(args.bootstrap_replicates)
    if n_bootstrap < 0:
        raise ValueError("--bootstrap-replicates must be nonnegative.")
    if args.l1_tolerance < 0:
        raise ValueError("--l1-tolerance must be nonnegative.")

    return Config(
        project_dir=project_dir,
        output_dir=output_dir,
        task_name=str(args.task_name),
        model=str(args.model),
        folds=tuple(FOLDS),
        v1_pattern=str(args.v1_pattern),
        v2_pattern=str(args.v2_pattern),
        v3_pattern=str(args.v3_pattern),
        v3_pred_key=str(args.v3_pred_key),
        v3_id_key=str(args.v3_id_key),
        l1_tolerance=float(args.l1_tolerance),
        bootstrap_replicates=n_bootstrap,
        bootstrap_seed=int(args.bootstrap_seed),
        make_plot=not bool(args.no_plot),
    )


def add_assertion(
    results: MutableSequence[AssertionResult],
    name: str,
    condition: bool,
    detail: str,
) -> bool:
    passed = bool(condition)
    results.append(AssertionResult(name=name, passed=passed, detail=detail))
    return passed


def require(
    results: MutableSequence[AssertionResult],
    name: str,
    condition: bool,
    detail: str,
) -> None:
    if not add_assertion(results, name, condition, detail):
        raise AssertionError(f"{name}: {detail}")


def normalize_id(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed / unavailable"


def resolve_pattern_path(
    project_dir: Path,
    pattern: str,
    *,
    fold: int,
    model: Optional[str] = None,
) -> Path:
    values = {"fold": fold}
    if model is not None:
        values["model"] = model
    raw = Path(pattern.format(**values)).expanduser()
    if not raw.is_absolute():
        raw = project_dir / raw
    return raw.resolve()


def format_float(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(
    rows: Iterable[Mapping[str, object]],
    columns: Sequence[Tuple[str, str]],
) -> str:
    rows = list(rows)
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                text = ""
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + body)


def load_official_oof_data(
    config: Config,
    assertions: MutableSequence[AssertionResult],
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], str]:
    try:
        from matbench.bench import MatbenchBenchmark
    except ImportError as exc:
        raise RuntimeError(
            "MatBench is required for a real run. Install the same MatBench environment "
            "used for the frozen predictions, or run --self-test only."
        ) from exc

    print("Loading official MatBench outer test folds (including targets)...")
    benchmark = MatbenchBenchmark(autoload=False)
    task = next(
        (candidate for candidate in benchmark.tasks if candidate.dataset_name == config.task_name),
        None,
    )
    if task is None:
        raise KeyError(f"MatBench task not found: {config.task_name}")
    task.load()
    target_col = str(task.metadata["target"])

    task_folds = tuple(int(fold) for fold in task.folds)
    require(
        assertions,
        "official fold labels match configured folds",
        set(task_folds) == set(config.folds),
        f"official={task_folds}; configured={config.folds}",
    )

    rows: List[Dict[str, object]] = []
    official_ids: Dict[int, np.ndarray] = {}

    for fold in config.folds:
        fold_df = task.get_test_data(fold, as_type="df", include_target=True)
        require(
            assertions,
            f"fold {fold} has a unique official index",
            bool(fold_df.index.is_unique),
            f"n_rows={len(fold_df)}; n_unique_ids={fold_df.index.nunique()}",
        )
        required_columns = {"structure", target_col}
        require(
            assertions,
            f"fold {fold} contains structure and target columns",
            required_columns.issubset(set(fold_df.columns)),
            f"required={sorted(required_columns)}; actual={list(fold_df.columns)}",
        )

        ids = np.asarray([normalize_id(value) for value in fold_df.index], dtype=object)
        official_ids[fold] = ids
        for position, (mbid, row) in enumerate(fold_df.iterrows()):
            structure = row["structure"]
            rows.append(
                {
                    "mbid": normalize_id(mbid),
                    "test_fold": int(fold),
                    "fold_position": int(position),
                    "reduced_formula": str(structure.composition.reduced_formula),
                    "y": float(row[target_col]),
                }
            )

    data = pd.DataFrame(rows)
    require(
        assertions,
        "every official OOF sample belongs to exactly one test fold",
        bool(data["mbid"].is_unique),
        f"n_rows={len(data)}; n_unique_mbid={data['mbid'].nunique()}",
    )
    require(
        assertions,
        "official labels are finite",
        bool(np.isfinite(data["y"].to_numpy(dtype=float)).all()),
        f"n_nonfinite={int((~np.isfinite(data['y'].to_numpy(dtype=float))).sum())}",
    )
    require(
        assertions,
        "official fold-position pairs are unique",
        not bool(data.duplicated(["test_fold", "fold_position"]).any()),
        f"n_duplicates={int(data.duplicated(['test_fold', 'fold_position']).sum())}",
    )

    data = data.sort_values(["test_fold", "fold_position"], kind="stable").reset_index(drop=True)
    print(
        f"Official OOF data: {len(data):,} samples, "
        f"{data['reduced_formula'].nunique():,} reduced formulas."
    )
    return data, official_ids, target_col


def add_global_formula_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    formula_group = result.groupby("reduced_formula", sort=False)["y"]
    result["formula_global_size"] = formula_group.transform("size").astype(int)
    result["global_formula_median_y"] = formula_group.transform("median")
    result["global_formula_spread_eV"] = formula_group.transform(
        lambda values: float(values.max() - values.min())
    )
    result["global_fixed_bound_ae"] = (
        result["y"] - result["global_formula_median_y"]
    ).abs()
    result["spread_bin"] = pd.cut(
        result["global_formula_spread_eV"],
        bins=list(SPREAD_EDGES),
        labels=list(SPREAD_LABELS),
        right=False,
        include_lowest=True,
        ordered=True,
    )
    if result["spread_bin"].isna().any():
        raise AssertionError("At least one nonnegative global formula spread was not assigned to a bin.")
    return result


def legacy_reproduction_gate(
    data: pd.DataFrame,
    output_dir: Path,
    assertions: MutableSequence[AssertionResult],
) -> pd.DataFrame:
    population_a = data.loc[data["formula_global_size"] >= 2].copy()
    n_samples = int(len(population_a))
    n_formulas = int(population_a["reduced_formula"].nunique())
    bound = float(population_a["global_fixed_bound_ae"].mean())
    bound_display = f"{bound:.4f}"

    print("\nLegacy compatibility reproduction:")
    print(f"  repeated-composition entries: {n_samples:,}")
    print(f"  same-formula groups:           {n_formulas:,}")
    print(f"  global fixed-function bound:   {bound:.8f} eV ({bound_display} eV at 4 d.p.)")

    checks = [
        (
            "legacy repeated-composition sample count",
            n_samples == LEGACY_REFERENCE_N_SAMPLES,
            f"computed={n_samples}; reference={LEGACY_REFERENCE_N_SAMPLES}",
        ),
        (
            "legacy same-formula group count",
            n_formulas == LEGACY_REFERENCE_N_FORMULAS,
            f"computed={n_formulas}; reference={LEGACY_REFERENCE_N_FORMULAS}",
        ),
        (
            "legacy global fixed-function bound at four decimals",
            bound_display == LEGACY_REFERENCE_BOUND_DISPLAY,
            f"computed={bound_display}; reference={LEGACY_REFERENCE_BOUND_DISPLAY}",
        ),
    ]
    for name, condition, detail in checks:
        add_assertion(assertions, name, condition, detail)

    if not all(condition for _, condition, _ in checks):
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic = pd.DataFrame(
            [
                {
                    "quantity": "repeated_composition_samples",
                    "computed": n_samples,
                    "reference": LEGACY_REFERENCE_N_SAMPLES,
                },
                {
                    "quantity": "same_formula_groups",
                    "computed": n_formulas,
                    "reference": LEGACY_REFERENCE_N_FORMULAS,
                },
                {
                    "quantity": "global_fixed_function_bound_eV",
                    "computed": bound,
                    "reference": LEGACY_REFERENCE_BOUND_DISPLAY,
                },
            ]
        )
        failure_path = output_dir / "legacy_reproduction_failure.csv"
        diagnostic.to_csv(failure_path, index=False)
        raise RuntimeError(
            "Legacy reproduction failed. New fold-conditioned conclusions were not computed. "
            f"See {failure_path}. Check MatBench/data versions and reduced-formula generation."
        )

    return population_a


def collect_prediction_paths(config: Config) -> Dict[str, Dict[int, Path]]:
    return {
        "L1": {
            fold: resolve_pattern_path(
                config.project_dir, config.v1_pattern, fold=fold, model=config.model
            )
            for fold in config.folds
        },
        "L2": {
            fold: resolve_pattern_path(
                config.project_dir, config.v2_pattern, fold=fold, model=config.model
            )
            for fold in config.folds
        },
        "L3": {
            fold: resolve_pattern_path(config.project_dir, config.v3_pattern, fold=fold)
            for fold in config.folds
        },
    }


def require_all_prediction_files(paths: Mapping[str, Mapping[int, Path]]) -> None:
    missing = [
        f"{level} fold {fold}: {path}"
        for level, fold_paths in paths.items()
        for fold, path in fold_paths.items()
        if not path.is_file()
    ]
    if missing:
        bullet_list = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "Required frozen clipped prediction files are missing:\n" + bullet_list
        )


def validate_numeric_predictions(
    values: np.ndarray,
    *,
    path: Path,
    expected_length: int,
    level: str,
    fold: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 1:
        array = raw
    elif raw.ndim == 2 and 1 in raw.shape:
        array = raw.reshape(-1)
    else:
        raise ValueError(
            f"{path}: expected a one-dimensional prediction array; actual shape={raw.shape}"
        )
    array = array.astype(np.float64, copy=False)
    if len(array) != expected_length:
        raise ValueError(
            f"{path}: {level} fold {fold} length={len(array)}; "
            f"official test length={expected_length}"
        )
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise ValueError(f"{path}: contains {bad} non-finite predictions.")
    if float(array.min(initial=0.0)) < -COMPARISON_TOLERANCE_EV:
        raise ValueError(
            f"{path}: contains negative predictions after the declared clipping step; "
            f"minimum={float(array.min()):.12g}"
        )
    return array


def load_position_bound_predictions(
    level: str,
    fold_paths: Mapping[int, Path],
    official_ids: Mapping[int, np.ndarray],
    input_records: MutableSequence[Dict[str, object]],
    assertions: MutableSequence[AssertionResult],
) -> pd.DataFrame:
    """
    Bind an ID-less legacy NPY array to official MatBench test IDs by fold position.

    This is an ingestion compatibility step backed by the reviewed L1/L2
    generation code.  After this function, all downstream joins use ``mbid``.
    """
    frames = []
    for fold, path in fold_paths.items():
        raw = np.load(path, allow_pickle=False)
        array = validate_numeric_predictions(
            raw,
            path=path,
            expected_length=len(official_ids[fold]),
            level=level,
            fold=fold,
        )
        frame = pd.DataFrame(
            {
                "mbid": official_ids[fold],
                "test_fold": int(fold),
                f"pred_{level}": array,
            }
        )
        frames.append(frame)
        input_records.append(
            {
                "level": level,
                "fold": fold,
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "mapping": "official MatBench test-fold position reconstructed from reviewed generator",
                "prediction_key": "",
                "id_key": "",
            }
        )

    result = pd.concat(frames, ignore_index=True)
    require(
        assertions,
        f"{level} reconstructed IDs are unique",
        bool(result["mbid"].is_unique),
        f"n_rows={len(result)}; n_unique_mbid={result['mbid'].nunique()}",
    )
    return result


def load_id_keyed_l3_predictions(
    config: Config,
    fold_paths: Mapping[int, Path],
    official_ids: Mapping[int, np.ndarray],
    input_records: MutableSequence[Dict[str, object]],
    assertions: MutableSequence[AssertionResult],
) -> pd.DataFrame:
    frames = []
    for fold, path in fold_paths.items():
        with np.load(path, allow_pickle=True) as archive:
            actual_keys = list(archive.files)
            required_keys = {config.v3_pred_key, config.v3_id_key}
            if not required_keys.issubset(set(actual_keys)):
                raise KeyError(
                    f"{path}: requires keys {sorted(required_keys)}; actual keys={actual_keys}"
                )
            ids = np.asarray(
                [normalize_id(value) for value in np.asarray(archive[config.v3_id_key]).reshape(-1)],
                dtype=object,
            )
            array = validate_numeric_predictions(
                np.asarray(archive[config.v3_pred_key]),
                path=path,
                expected_length=len(ids),
                level="L3",
                fold=fold,
            )

        if len(ids) != len(np.unique(ids)):
            duplicated = pd.Series(ids)[pd.Series(ids).duplicated(keep=False)].unique().tolist()
            raise ValueError(f"{path}: duplicate L3 IDs found; examples={duplicated[:10]}")

        official = official_ids[fold]
        missing = sorted(set(official) - set(ids))
        extra = sorted(set(ids) - set(official))
        require(
            assertions,
            f"L3 fold {fold} ID set matches the official test fold",
            not missing and not extra and len(ids) == len(official),
            f"n_ids={len(ids)}; official={len(official)}; missing={len(missing)}; extra={len(extra)}",
        )

        frames.append(
            pd.DataFrame(
                {
                    "mbid": ids,
                    "test_fold": int(fold),
                    "pred_L3": array,
                }
            )
        )
        input_records.append(
            {
                "level": "L3",
                "fold": fold,
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "mapping": "direct ID join from NPZ",
                "prediction_key": config.v3_pred_key,
                "id_key": config.v3_id_key,
            }
        )

    result = pd.concat(frames, ignore_index=True)
    require(
        assertions,
        "L3 IDs are unique across all official folds",
        bool(result["mbid"].is_unique),
        f"n_rows={len(result)}; n_unique_mbid={result['mbid'].nunique()}",
    )
    return result


def attach_predictions_by_id(
    data: pd.DataFrame,
    prediction_frames: Sequence[pd.DataFrame],
    assertions: MutableSequence[AssertionResult],
) -> pd.DataFrame:
    result = data.copy()
    official_ids = set(result["mbid"])

    for frame in prediction_frames:
        pred_col = next(column for column in frame.columns if column.startswith("pred_"))
        level = pred_col.replace("pred_", "")
        prediction_ids = set(frame["mbid"])
        missing = official_ids - prediction_ids
        extra = prediction_ids - official_ids
        require(
            assertions,
            f"{level} covers exactly the official OOF sample IDs",
            not missing and not extra and len(frame) == len(result),
            (
                f"official={len(result)}; predictions={len(frame)}; "
                f"missing={len(missing)}; extra={len(extra)}"
            ),
        )

        fold_check = result[["mbid", "test_fold"]].merge(
            frame[["mbid", "test_fold"]],
            on="mbid",
            how="outer",
            suffixes=("_official", "_prediction"),
            validate="one_to_one",
            indicator=True,
        )
        fold_mismatch = fold_check.loc[
            (fold_check["_merge"] != "both")
            | (fold_check["test_fold_official"] != fold_check["test_fold_prediction"])
        ]
        require(
            assertions,
            f"{level} fold labels agree with the official mapping",
            fold_mismatch.empty,
            f"n_mismatches={len(fold_mismatch)}",
        )

        result = result.merge(
            frame[["mbid", pred_col]],
            on="mbid",
            how="left",
            validate="one_to_one",
        )
        require(
            assertions,
            f"{level} has no missing values after stable-ID merge",
            not bool(result[pred_col].isna().any()),
            f"n_missing={int(result[pred_col].isna().sum())}",
        )

    required_prediction_columns = {"pred_L1", "pred_L2", "pred_L3"}
    require(
        assertions,
        "labels and all three prediction levels cover the same samples",
        required_prediction_columns.issubset(result.columns)
        and not bool(result[list(required_prediction_columns)].isna().any().any()),
        f"prediction_columns={sorted(column for column in result if column.startswith('pred_'))}",
    )

    for level in ("L1", "L2", "L3"):
        result[f"ae_{level}"] = (result["y"] - result[f"pred_{level}"]).abs()
    return result


def derive_fold_conditioned_populations(
    population_a: pd.DataFrame,
    assertions: MutableSequence[AssertionResult],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    population_a = population_a.copy()
    fold_group = population_a.groupby(
        ["reduced_formula", "test_fold"], sort=False
    )["y"]
    population_a["formula_fold_size"] = fold_group.transform("size").astype(int)
    population_a["formula_fold_median_y"] = fold_group.transform("median")
    population_a["fold_conditioned_bound_ae"] = (
        population_a["y"] - population_a["formula_fold_median_y"]
    ).abs()

    singleton = population_a["formula_fold_size"] == 1
    singleton_max = (
        float(population_a.loc[singleton, "fold_conditioned_bound_ae"].max())
        if singleton.any()
        else 0.0
    )
    require(
        assertions,
        "size=1 (formula, fold) groups contribute exactly zero oracle error",
        singleton_max == 0.0,
        f"n_singleton_samples={int(singleton.sum())}; max_contribution={singleton_max:.12g}",
    )

    global_bound = float(population_a["global_fixed_bound_ae"].mean())
    fold_bound = float(population_a["fold_conditioned_bound_ae"].mean())
    require(
        assertions,
        "fold-conditioned oracle is no larger than the global fixed-function oracle",
        fold_bound <= global_bound + COMPARISON_TOLERANCE_EV,
        f"fold_conditioned={fold_bound:.12g}; global_fixed={global_bound:.12g}",
    )

    population_b = population_a.loc[population_a["formula_fold_size"] >= 2].copy()
    require(
        assertions,
        "same-fold repeated subset contains only complete size>=2 groups",
        not population_b.empty
        and bool((population_b["formula_fold_size"] >= 2).all()),
        (
            f"n_samples={len(population_b)}; "
            f"min_group_size={population_b['formula_fold_size'].min() if len(population_b) else 'NA'}"
        ),
    )
    require(
        assertions,
        "same-fold repeated subset is a subset of the global repeated-formula subset",
        set(population_b["mbid"]).issubset(set(population_a["mbid"])),
        f"A={len(population_a)}; B={len(population_b)}",
    )
    return population_a, population_b


def l1_constancy_table(
    population_a: pd.DataFrame,
    tolerance: float,
    assertions: MutableSequence[AssertionResult],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    grouped = (
        population_a.groupby(["reduced_formula", "test_fold"], sort=True)["pred_L1"]
        .agg(
            group_size="size",
            prediction_min="min",
            prediction_max="max",
            prediction_std=lambda values: float(np.std(values.to_numpy(dtype=float), ddof=0)),
        )
        .reset_index()
    )
    grouped = grouped.loc[grouped["group_size"] >= 2].copy()
    grouped["prediction_range"] = grouped["prediction_max"] - grouped["prediction_min"]
    grouped["strictly_nonzero"] = (
        (grouped["prediction_range"] != 0.0) | (grouped["prediction_std"] != 0.0)
    )
    grouped["exceeds_tolerance"] = (
        (grouped["prediction_range"] > tolerance)
        | (grouped["prediction_std"] > tolerance)
    )

    exceptions = grouped.loc[grouped["strictly_nonzero"]].sort_values(
        ["prediction_range", "prediction_std"], ascending=False
    )
    stats = {
        "n_groups_checked": int(len(grouped)),
        "max_range_eV": float(grouped["prediction_range"].max()) if len(grouped) else 0.0,
        "max_std_eV": float(grouped["prediction_std"].max()) if len(grouped) else 0.0,
        "n_strictly_nonzero_groups": int(grouped["strictly_nonzero"].sum()),
        "n_groups_exceeding_tolerance": int(grouped["exceeds_tolerance"].sum()),
        "tolerance_eV": float(tolerance),
    }
    add_assertion(
        assertions,
        "L1 is constant within every size>=2 (formula, fold) group at tolerance",
        stats["n_groups_exceeding_tolerance"] == 0,
        (
            f"checked={stats['n_groups_checked']}; "
            f"max_range={stats['max_range_eV']:.12g}; "
            f"max_std={stats['max_std_eV']:.12g}; "
            f"tolerance={tolerance:.12g}; "
            f"n_exceeding={stats['n_groups_exceeding_tolerance']}"
        ),
    )
    add_assertion(
        assertions,
        "L1 is exactly constant within every size>=2 (formula, fold) group",
        stats["n_strictly_nonzero_groups"] == 0,
        f"n_strictly_nonzero={stats['n_strictly_nonzero_groups']}",
    )
    return exceptions, stats


def group_size_counts(frame: pd.DataFrame) -> Dict[str, int]:
    sizes = frame.groupby(["reduced_formula", "test_fold"], sort=False).size()
    singleton = sizes == 1
    repeated = sizes >= 2
    return {
        "n_formula_fold_groups": int(len(sizes)),
        "n_size1_groups": int(singleton.sum()),
        "n_size1_samples": int(sizes.loc[singleton].sum()),
        "n_size_ge2_groups": int(repeated.sum()),
        "n_size_ge2_samples": int(sizes.loc[repeated].sum()),
    }


def comparison_fields(
    prefix: str,
    model_mae: float,
    bound: float,
) -> Dict[str, object]:
    delta = float(model_mae - bound)
    undercut = np.nan
    above = np.nan
    if delta < -COMPARISON_TOLERANCE_EV:
        relation = "undercut"
        if bound > 0:
            undercut = 100.0 * (bound - model_mae) / bound
    elif delta > COMPARISON_TOLERANCE_EV:
        relation = "above_bound"
        if bound > 0:
            above = 100.0 * (model_mae - bound) / bound
    else:
        relation = "equal_within_tolerance"
    return {
        f"{prefix}_minus_bound_eV": delta,
        f"{prefix}_relation": relation,
        f"{prefix}_undercut_percent": undercut,
        f"{prefix}_above_bound_percent": above,
    }


def summarize_frame(
    frame: pd.DataFrame,
    *,
    analysis: str,
    population: str,
    bound_definition: str,
    bound_column: str,
    fold: object,
) -> Dict[str, object]:
    counts = group_size_counts(frame)
    bound = float(frame[bound_column].mean()) if len(frame) else np.nan
    l1 = float(frame["ae_L1"].mean()) if len(frame) else np.nan
    l2 = float(frame["ae_L2"].mean()) if len(frame) else np.nan
    l3 = float(frame["ae_L3"].mean()) if len(frame) else np.nan
    row: Dict[str, object] = {
        "analysis": analysis,
        "population": population,
        "fold": fold,
        "bound_definition": bound_definition,
        "n_samples": int(len(frame)),
        "n_formulas": int(frame["reduced_formula"].nunique()),
        **counts,
        "oracle_bound_eV": bound,
        "L1_MAE_eV": l1,
        "L2_MAE_eV": l2,
        "L3_MAE_eV": l3,
    }
    row.update(comparison_fields("L2", l2, bound))
    row.update(comparison_fields("L3", l3, bound))
    return row


def build_summary(
    population_a: pd.DataFrame,
    population_b: pd.DataFrame,
    folds: Sequence[int],
) -> pd.DataFrame:
    rows = [
        summarize_frame(
            population_a,
            analysis="legacy_reproduction",
            population=POPULATION_A,
            bound_definition="global_fixed_function_oracle",
            bound_column="global_fixed_bound_ae",
            fold="pooled",
        ),
        summarize_frame(
            population_a,
            analysis="fold_conditioned",
            population=POPULATION_A,
            bound_definition="fold_conditioned_oof_oracle",
            bound_column="fold_conditioned_bound_ae",
            fold="pooled",
        ),
        summarize_frame(
            population_b,
            analysis="fold_conditioned",
            population=POPULATION_B,
            bound_definition="fold_conditioned_oof_oracle",
            bound_column="fold_conditioned_bound_ae",
            fold="pooled",
        ),
    ]
    for population_name, frame in (
        (POPULATION_A, population_a),
        (POPULATION_B, population_b),
    ):
        for fold in folds:
            rows.append(
                summarize_frame(
                    frame.loc[frame["test_fold"] == fold],
                    analysis="fold_conditioned",
                    population=population_name,
                    bound_definition="fold_conditioned_oof_oracle",
                    bound_column="fold_conditioned_bound_ae",
                    fold=int(fold),
                )
            )
    return pd.DataFrame(rows)


def build_group_size_distribution(
    population_a: pd.DataFrame,
    population_b: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for population, frame in (
        (POPULATION_A, population_a),
        (POPULATION_B, population_b),
    ):
        sizes = frame.groupby(["reduced_formula", "test_fold"], sort=False).size()
        for group_size, n_groups in sizes.value_counts().sort_index().items():
            n_samples = int(group_size * n_groups)
            rows.append(
                {
                    "population": population,
                    "formula_fold_group_size": int(group_size),
                    "n_groups": int(n_groups),
                    "n_samples": n_samples,
                    "fraction_of_groups": float(n_groups / len(sizes)),
                    "fraction_of_samples": float(n_samples / len(frame)),
                }
            )
    return pd.DataFrame(rows)


def spread_row(
    frame: pd.DataFrame,
    *,
    analysis: str,
    population: str,
    spread_bin: str,
    bound_definition: str,
    bound_column: str,
) -> Dict[str, object]:
    if len(frame):
        bound = float(frame[bound_column].mean())
        l1 = float(frame["ae_L1"].mean())
        l2 = float(frame["ae_L2"].mean())
        l3 = float(frame["ae_L3"].mean())
        n_groups = int(
            frame[["reduced_formula", "test_fold"]].drop_duplicates().shape[0]
        )
    else:
        bound = l1 = l2 = l3 = np.nan
        n_groups = 0
    row: Dict[str, object] = {
        "analysis": analysis,
        "population": population,
        "spread_definition": "global within-formula label spread",
        "spread_bin": spread_bin,
        "n_samples": int(len(frame)),
        "n_formulas": int(frame["reduced_formula"].nunique()),
        "n_formula_fold_groups": n_groups,
        "bound_definition": bound_definition,
        "oracle_bound_eV": bound,
        "L1_MAE_eV": l1,
        "L2_MAE_eV": l2,
        "L3_MAE_eV": l3,
    }
    if len(frame):
        row.update(comparison_fields("L2", l2, bound))
        row.update(comparison_fields("L3", l3, bound))
    else:
        for prefix in ("L2", "L3"):
            row.update(
                {
                    f"{prefix}_minus_bound_eV": np.nan,
                    f"{prefix}_relation": "",
                    f"{prefix}_undercut_percent": np.nan,
                    f"{prefix}_above_bound_percent": np.nan,
                }
            )
    return row


def build_spread_table(
    population_a: pd.DataFrame,
    population_b: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for spread_label in SPREAD_LABELS:
        legacy_sub = population_a.loc[population_a["spread_bin"] == spread_label]
        rows.append(
            spread_row(
                legacy_sub,
                analysis="legacy_reproduction",
                population=POPULATION_A,
                spread_bin=spread_label,
                bound_definition="global_fixed_function_oracle",
                bound_column="global_fixed_bound_ae",
            )
        )
    for population, frame in (
        (POPULATION_A, population_a),
        (POPULATION_B, population_b),
    ):
        for spread_label in SPREAD_LABELS:
            sub = frame.loc[frame["spread_bin"] == spread_label]
            rows.append(
                spread_row(
                    sub,
                    analysis="fold_conditioned",
                    population=population,
                    spread_bin=spread_label,
                    bound_definition="fold_conditioned_oof_oracle",
                    bound_column="fold_conditioned_bound_ae",
                )
            )
    return pd.DataFrame(rows)


def formula_cluster_sufficient_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse each formula once, before resampling, into additive statistics."""
    return (
        frame.groupby("reduced_formula", sort=True)
        .agg(
            n_samples=("mbid", "size"),
            bound_sum=("fold_conditioned_bound_ae", "sum"),
            l2_ae_sum=("ae_L2", "sum"),
            l3_ae_sum=("ae_L3", "sum"),
        )
        .reset_index()
    )


def resampled_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    draw: np.ndarray,
) -> float:
    """
    Return a sample-weighted statistic for one cluster-bootstrap draw.

    ``draw`` is intentionally used as an integer index array without applying
    ``unique`` or a post-draw groupby. If a formula index occurs m times, its
    numerator and sample count therefore occur m times in both sums.
    """
    return float(numerator[draw].sum()) / float(denominator[draw].sum())


def formula_cluster_bootstrap_statistics(
    frame: pd.DataFrame,
    *,
    n_replicates: int,
    seed: int,
) -> Dict[str, object]:
    """
    Bootstrap three sample-weighted metrics with ``reduced_formula`` clusters.

    Every formula's entries across all folds in the supplied analysis subset
    enter together. Formula multiplicity from sampling with replacement is
    retained by direct repeated integer indexing.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive for bootstrap statistics.")
    if frame.empty:
        raise ValueError("Cannot bootstrap an empty analysis subset.")

    aggregates = formula_cluster_sufficient_statistics(frame)
    n_clusters = len(aggregates)
    cluster_n = aggregates["n_samples"].to_numpy(dtype=np.float64)
    bound_sum = aggregates["bound_sum"].to_numpy(dtype=np.float64)
    metric_numerators = {
        "fold_conditioned_bound_eV": bound_sum,
        "L2_MAE_minus_bound_eV": (
            aggregates["l2_ae_sum"].to_numpy(dtype=np.float64) - bound_sum
        ),
        "L3_MAE_minus_bound_eV": (
            aggregates["l3_ae_sum"].to_numpy(dtype=np.float64) - bound_sum
        ),
    }

    # Point estimates use the same additive sufficient statistics as the
    # bootstrap and are therefore exactly sample-weighted on the chosen subset.
    denominator = float(cluster_n.sum())
    point_estimates = {
        metric: float(numerator.sum()) / denominator
        for metric, numerator in metric_numerators.items()
    }

    rng = np.random.default_rng(seed)
    estimates = {
        metric: np.empty(n_replicates, dtype=np.float64)
        for metric in metric_numerators
    }
    for replicate in range(n_replicates):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        for metric, numerator in metric_numerators.items():
            estimates[metric][replicate] = resampled_ratio(
                numerator, cluster_n, draw
            )

    intervals = {
        metric: tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))
        for metric, values in estimates.items()
    }
    return {
        "point_estimates": point_estimates,
        "intervals": intervals,
        "n_clusters": int(n_clusters),
        "n_samples": int(len(frame)),
        "n_formula_fold_groups": int(
            frame[["reduced_formula", "test_fold"]].drop_duplicates().shape[0]
        ),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    population: str,
    n_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Pooled formula-cluster bootstrap retained for the original two populations."""
    if n_replicates <= 0:
        return pd.DataFrame()

    statistics = formula_cluster_bootstrap_statistics(
        frame, n_replicates=n_replicates, seed=seed
    )
    rows = []
    for metric, point_estimate in statistics["point_estimates"].items():
        lower, upper = statistics["intervals"][metric]
        rows.append(
            {
                "population": population,
                "metric": metric,
                "point_estimate": point_estimate,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "n_bootstrap": int(n_replicates),
                "seed": int(seed),
                "cluster_unit": "reduced_formula",
                "n_unique_formula_clusters": statistics["n_clusters"],
                "multiplicity_preserved": True,
            }
        )
    return pd.DataFrame(rows)


def ci_zero_classification(lower: float, upper: float) -> str:
    """Classify the pointwise CI for model MAE minus the empirical oracle."""
    if upper < 0.0:
        return "robust_undercut_ci_upper_below_zero"
    if lower > 0.0:
        return "robust_above_bound_ci_lower_above_zero"
    return "inconclusive_ci_includes_zero"


def build_spread_bootstrap_table(
    population_a: pd.DataFrame,
    population_b: pd.DataFrame,
    *,
    n_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """
    Formula-cluster bootstrap within global within-formula spread bins.

    A formula has one global spread and hence belongs to exactly one bin. Within
    each population/bin, all available folds and samples of a sampled formula
    remain together. Each bin initializes a deterministic RNG with the recorded
    root seed; repeated formula indices retain their sampling multiplicity.
    """
    columns = [
        "population",
        "spread_definition",
        "spread_bin",
        "n_samples",
        "n_formulas",
        "n_formula_fold_groups",
        "oracle_bound_eV",
        "oracle_bound_ci_95_lower",
        "oracle_bound_ci_95_upper",
        "L2_MAE_minus_bound_eV",
        "L2_ci_95_lower",
        "L2_ci_95_upper",
        "L2_ci_zero_classification",
        "L3_MAE_minus_bound_eV",
        "L3_ci_95_lower",
        "L3_ci_95_upper",
        "L3_ci_zero_classification",
        "n_bootstrap",
        "seed",
        "interval_method",
        "cluster_unit",
        "multiplicity_preserved",
    ]
    if n_replicates <= 0:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, object]] = []
    for population, frame in (
        (POPULATION_A, population_a),
        (POPULATION_B, population_b),
    ):
        formula_bin_counts = frame.groupby("reduced_formula", sort=False)[
            "spread_bin"
        ].nunique()
        if not bool((formula_bin_counts == 1).all()):
            raise AssertionError(
                f"At least one formula maps to multiple global spread bins in {population}."
            )

        for spread_label in SPREAD_LABELS:
            subset = frame.loc[frame["spread_bin"] == spread_label].copy()
            if subset.empty:
                rows.append(
                    {
                        "population": population,
                        "spread_definition": "global within-formula label spread",
                        "spread_bin": spread_label,
                        "n_samples": 0,
                        "n_formulas": 0,
                        "n_formula_fold_groups": 0,
                        "oracle_bound_eV": np.nan,
                        "oracle_bound_ci_95_lower": np.nan,
                        "oracle_bound_ci_95_upper": np.nan,
                        "L2_MAE_minus_bound_eV": np.nan,
                        "L2_ci_95_lower": np.nan,
                        "L2_ci_95_upper": np.nan,
                        "L2_ci_zero_classification": "empty_bin",
                        "L3_MAE_minus_bound_eV": np.nan,
                        "L3_ci_95_lower": np.nan,
                        "L3_ci_95_upper": np.nan,
                        "L3_ci_zero_classification": "empty_bin",
                        "n_bootstrap": int(n_replicates),
                        "seed": int(seed),
                        "interval_method": "percentile_2.5_97.5",
                        "cluster_unit": "reduced_formula",
                        "multiplicity_preserved": True,
                    }
                )
                continue

            statistics = formula_cluster_bootstrap_statistics(
                subset, n_replicates=n_replicates, seed=seed
            )
            points = statistics["point_estimates"]
            intervals = statistics["intervals"]
            bound_lower, bound_upper = intervals["fold_conditioned_bound_eV"]
            l2_lower, l2_upper = intervals["L2_MAE_minus_bound_eV"]
            l3_lower, l3_upper = intervals["L3_MAE_minus_bound_eV"]
            rows.append(
                {
                    "population": population,
                    "spread_definition": "global within-formula label spread",
                    "spread_bin": spread_label,
                    "n_samples": statistics["n_samples"],
                    "n_formulas": statistics["n_clusters"],
                    "n_formula_fold_groups": statistics["n_formula_fold_groups"],
                    "oracle_bound_eV": points["fold_conditioned_bound_eV"],
                    "oracle_bound_ci_95_lower": bound_lower,
                    "oracle_bound_ci_95_upper": bound_upper,
                    "L2_MAE_minus_bound_eV": points["L2_MAE_minus_bound_eV"],
                    "L2_ci_95_lower": l2_lower,
                    "L2_ci_95_upper": l2_upper,
                    "L2_ci_zero_classification": ci_zero_classification(
                        l2_lower, l2_upper
                    ),
                    "L3_MAE_minus_bound_eV": points["L3_MAE_minus_bound_eV"],
                    "L3_ci_95_lower": l3_lower,
                    "L3_ci_95_upper": l3_upper,
                    "L3_ci_zero_classification": ci_zero_classification(
                        l3_lower, l3_upper
                    ),
                    "n_bootstrap": int(n_replicates),
                    "seed": int(seed),
                    "interval_method": "percentile_2.5_97.5",
                    "cluster_unit": "reduced_formula",
                    "multiplicity_preserved": True,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def validate_spread_bootstrap_table(
    spread_bootstrap: pd.DataFrame,
    spread_table: pd.DataFrame,
    *,
    config: Config,
    assertions: MutableSequence[AssertionResult],
) -> None:
    """Audit bin coverage, point estimates, fixed settings, and reference values."""
    add_assertion(
        assertions,
        "spread-bin bootstrap uses 2,000 replicates and seed 20260730",
        config.bootstrap_replicates == BOOTSTRAP_REPLICATES
        and config.bootstrap_seed == BOOTSTRAP_SEED,
        (
            f"replicates={config.bootstrap_replicates}; seed={config.bootstrap_seed}; "
            f"required_replicates={BOOTSTRAP_REPLICATES}; required_seed={BOOTSTRAP_SEED}"
        ),
    )
    add_assertion(
        assertions,
        "spread-bin bootstrap explicitly uses global within-formula label spread",
        len(spread_bootstrap) > 0
        and bool(
            (
                spread_bootstrap["spread_definition"]
                == "global within-formula label spread"
            ).all()
        ),
        f"definitions={sorted(spread_bootstrap['spread_definition'].dropna().unique()) if len(spread_bootstrap) else []}",
    )
    add_assertion(
        assertions,
        "spread-bin bootstrap preserves repeated formula sampling multiplicity",
        len(spread_bootstrap) > 0
        and bool(spread_bootstrap["multiplicity_preserved"].all()),
        "implementation uses repeated integer indices with no post-draw groupby or deduplication",
    )

    main_spread = spread_table.loc[
        spread_table["analysis"] == "fold_conditioned"
    ].copy()
    point_match_failures = []
    for _, row in spread_bootstrap.iterrows():
        if int(row["n_samples"]) == 0:
            point_match_failures.append(
                f"{row['population']} / {row['spread_bin']}: empty bin"
            )
            continue
        expected_rows = main_spread.loc[
            (main_spread["population"] == row["population"])
            & (main_spread["spread_bin"] == row["spread_bin"])
        ]
        if len(expected_rows) != 1:
            point_match_failures.append(
                f"{row['population']} / {row['spread_bin']}: main spread row count={len(expected_rows)}"
            )
            continue
        expected = expected_rows.iloc[0]
        comparisons = {
            "oracle_bound_eV": "oracle_bound_eV",
            "L2_MAE_minus_bound_eV": "L2_minus_bound_eV",
            "L3_MAE_minus_bound_eV": "L3_minus_bound_eV",
        }
        for bootstrap_column, spread_column in comparisons.items():
            if f"{float(row[bootstrap_column]):.12f}" != f"{float(expected[spread_column]):.12f}":
                point_match_failures.append(
                    f"{row['population']} / {row['spread_bin']} / {bootstrap_column}: "
                    f"bootstrap={float(row[bootstrap_column]):.12f}, "
                    f"spread={float(expected[spread_column]):.12f}"
                )
    add_assertion(
        assertions,
        "spread-bin bootstrap point estimates equal the existing spread-table estimates",
        not point_match_failures,
        "all rows match at 12 decimals"
        if not point_match_failures
        else "; ".join(point_match_failures[:10]),
    )

    same_fold = spread_bootstrap.loc[
        spread_bootstrap["population"] == POPULATION_B
    ].set_index("spread_bin")
    add_assertion(
        assertions,
        "same-fold repeated subset has all four required global-spread bins",
        set(same_fold.index) == set(SPREAD_LABELS)
        and bool((same_fold["n_samples"] > 0).all()),
        (
            f"bins={list(same_fold.index)}; "
            f"sample_counts={same_fold['n_samples'].to_dict()}"
        ),
    )

    for spread_label, references in SAME_FOLD_SPREAD_DELTA_REFERENCE.items():
        if spread_label not in same_fold.index:
            for metric, reference in references.items():
                add_assertion(
                    assertions,
                    f"acceptance point estimate {POPULATION_B} / {spread_label} / {metric}",
                    False,
                    f"bin missing; reference={reference:+.6f}",
                )
            continue
        row = same_fold.loc[spread_label]
        for metric, reference in references.items():
            actual = float(row[metric])
            add_assertion(
                assertions,
                f"acceptance point estimate {POPULATION_B} / {spread_label} / {metric}",
                f"{actual:.6f}" == f"{reference:.6f}",
                f"computed={actual:+.6f}; reference={reference:+.6f}",
            )


def make_spread_plot(spread_table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    main = spread_table.loc[spread_table["analysis"] == "fold_conditioned"].copy()
    populations = (POPULATION_A, POPULATION_B)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {
        "oracle_bound_eV": "#AAAAAA",
        "L1_MAE_eV": "#85B7EB",
        "L2_MAE_eV": "#185FA5",
        "L3_MAE_eV": "#C0392B",
    }
    labels = {
        "oracle_bound_eV": "Fold-conditioned oracle bound",
        "L1_MAE_eV": "Level 1",
        "L2_MAE_eV": "Level 2",
        "L3_MAE_eV": "Level 3",
    }

    for axis, population in zip(axes, populations):
        sub = main.loc[main["population"] == population].set_index("spread_bin")
        sub = sub.reindex(SPREAD_LABELS)
        x = np.arange(len(SPREAD_LABELS), dtype=float)
        columns = ("oracle_bound_eV", "L1_MAE_eV", "L2_MAE_eV", "L3_MAE_eV")
        width = 0.8 / len(columns)
        for index, column in enumerate(columns):
            axis.bar(
                x + (index - (len(columns) - 1) / 2) * width,
                sub[column].to_numpy(dtype=float),
                width,
                label=labels[column],
                color=colors[column],
                edgecolor="white",
            )
        sample_counts = sub["n_samples"].fillna(0).astype(int).tolist()
        axis.set_xticks(x)
        axis.set_xticklabels(
            [f"{label}\n(n={count:,})" for label, count in zip(SPREAD_LABELS, sample_counts)]
        )
        axis.set_xlabel("Global within-formula label spread")
        axis.set_title(population.replace("_", " "))
        axis.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Sample-weighted MAE (eV)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Fold-conditioned composition oracle vs frozen clipped OOF errors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_spread_ci_plot(
    spread_bootstrap: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot L2/L3 MAE minus the fold-conditioned oracle with percentile CIs.

    The inferential panel is intentionally restricted to
    ``same_fold_repeated_subset``, where every retained (formula, fold) group
    has size >= 2. CIs are drawn from the paired, formula-cluster bootstrap of
    the difference itself rather than by subtracting two marginal intervals.
    """
    import matplotlib.pyplot as plt

    required_columns = {
        "population",
        "spread_bin",
        "n_samples",
        "L2_MAE_minus_bound_eV",
        "L2_ci_95_lower",
        "L2_ci_95_upper",
        "L2_ci_zero_classification",
        "L3_MAE_minus_bound_eV",
        "L3_ci_95_lower",
        "L3_ci_95_upper",
        "L3_ci_zero_classification",
        "n_bootstrap",
        "seed",
    }
    missing = sorted(required_columns - set(spread_bootstrap.columns))
    if missing:
        raise KeyError(f"Spread-bootstrap plot is missing columns: {missing}")

    subset = spread_bootstrap.loc[
        spread_bootstrap["population"] == POPULATION_B
    ].copy()
    if subset["spread_bin"].duplicated().any():
        duplicates = subset.loc[
            subset["spread_bin"].duplicated(keep=False), "spread_bin"
        ].tolist()
        raise ValueError(f"Duplicate same-fold spread-bootstrap rows: {duplicates}")
    subset = subset.set_index("spread_bin").reindex(SPREAD_LABELS)
    if subset["n_samples"].isna().any() or bool((subset["n_samples"] <= 0).any()):
        raise ValueError(
            "CI plot requires one nonempty same-fold repeated-subset row for every "
            f"global spread bin; sample counts={subset['n_samples'].to_dict()}"
        )

    bootstrap_values = subset["n_bootstrap"].dropna().astype(int).unique()
    seed_values = subset["seed"].dropna().astype(int).unique()
    if len(bootstrap_values) != 1 or len(seed_values) != 1:
        raise ValueError(
            "CI plot rows must share one bootstrap configuration; "
            f"replicates={bootstrap_values.tolist()}, seeds={seed_values.tolist()}"
        )

    x = np.arange(len(SPREAD_LABELS), dtype=float)
    offset = 0.11
    cap_half_width = 0.045
    series = (
        {
            "name": "Level 2",
            "color": "#185FA5",
            "marker": "o",
            "x": x - offset,
            "point": subset["L2_MAE_minus_bound_eV"].to_numpy(dtype=float),
            "lower": subset["L2_ci_95_lower"].to_numpy(dtype=float),
            "upper": subset["L2_ci_95_upper"].to_numpy(dtype=float),
            "classification": subset["L2_ci_zero_classification"].astype(str).tolist(),
        },
        {
            "name": "Level 3",
            "color": "#C0392B",
            "marker": "s",
            "x": x + offset,
            "point": subset["L3_MAE_minus_bound_eV"].to_numpy(dtype=float),
            "lower": subset["L3_ci_95_lower"].to_numpy(dtype=float),
            "upper": subset["L3_ci_95_upper"].to_numpy(dtype=float),
            "classification": subset["L3_ci_zero_classification"].astype(str).tolist(),
        },
    )

    finite_values = np.concatenate(
        [
            np.concatenate([item["point"], item["lower"], item["upper"]])
            for item in series
        ]
    )
    if not np.isfinite(finite_values).all():
        raise ValueError("CI plot received non-finite point estimates or interval limits.")
    y_min = min(float(finite_values.min()), 0.0)
    y_max = max(float(finite_values.max()), 0.0)
    span = max(y_max - y_min, 0.05)
    y_min -= 0.14 * span
    y_max += 0.18 * span

    fig, axis = plt.subplots(figsize=(10.2, 5.8))
    axis.set_ylim(y_min, y_max)
    axis.axhspan(y_min, 0.0, color="#E8F3EC", alpha=0.55, zorder=0)
    axis.axhspan(0.0, y_max, color="#F7F3EA", alpha=0.45, zorder=0)
    axis.axhline(
        0.0,
        color="#333333",
        linewidth=1.3,
        linestyle="--",
        zorder=2,
        label="Oracle equality (difference = 0)",
    )

    for item in series:
        axis.vlines(
            item["x"],
            item["lower"],
            item["upper"],
            color=item["color"],
            linewidth=2.0,
            zorder=3,
        )
        axis.hlines(
            item["lower"],
            item["x"] - cap_half_width,
            item["x"] + cap_half_width,
            color=item["color"],
            linewidth=1.7,
            zorder=3,
        )
        axis.hlines(
            item["upper"],
            item["x"] - cap_half_width,
            item["x"] + cap_half_width,
            color=item["color"],
            linewidth=1.7,
            zorder=3,
        )
        axis.scatter(
            item["x"],
            item["point"],
            s=55,
            marker=item["marker"],
            color=item["color"],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label=f"{item['name']} point estimate and 95% CI",
        )
        for x_value, lower, classification in zip(
            item["x"], item["lower"], item["classification"]
        ):
            if classification == "robust_undercut_ci_upper_below_zero":
                axis.annotate(
                    "robust undercut",
                    xy=(x_value, lower),
                    xytext=(0, -6),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    color=item["color"],
                )

    sample_counts = subset["n_samples"].astype(int).tolist()
    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            f"{label}\n(n={count:,})"
            for label, count in zip(SPREAD_LABELS, sample_counts)
        ]
    )
    axis.set_xlabel("Global within-formula label spread")
    axis.set_ylabel("OOF MAE − fold-conditioned oracle bound (eV)")
    axis.set_title(
        "Same-fold repeated-composition entries: model error relative to oracle"
    )
    axis.text(
        0.01,
        0.02,
        "Negative: below oracle bound",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#355E45",
        va="bottom",
    )
    axis.text(
        0.01,
        0.98,
        "Positive: above oracle bound",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#705C32",
        va="top",
    )
    axis.grid(axis="y", alpha=0.25, zorder=1)
    axis.legend(loc="upper right", fontsize=8.5, frameon=True)
    fig.text(
        0.5,
        0.005,
        (
            "95% percentile formula-cluster bootstrap CI; cluster = reduced_formula; "
            f"replicates = {int(bootstrap_values[0]):,}; seed = {int(seed_values[0])}"
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def relation_sentence(model: str, row: Mapping[str, object]) -> str:
    bound = float(row["oracle_bound_eV"])
    mae = float(row[f"{model}_MAE_eV"])
    delta = float(row[f"{model}_minus_bound_eV"])
    relation = str(row[f"{model}_relation"])
    if relation == "undercut":
        percent = row[f"{model}_undercut_percent"]
        return (
            f"{model} MAE {mae:.6f} eV is below the bound {bound:.6f} eV "
            f"by {-delta:.6f} eV ({float(percent):.2f}% undercut)."
        )
    if relation == "above_bound":
        percent = row[f"{model}_above_bound_percent"]
        percent_text = (
            f", {float(percent):.2f}% above" if percent is not None and np.isfinite(percent) else ""
        )
        return (
            f"{model} MAE {mae:.6f} eV is above the bound {bound:.6f} eV "
            f"by {delta:.6f} eV{percent_text}."
        )
    return (
        f"{model} MAE {mae:.6f} eV equals the bound {bound:.6f} eV "
        f"within {COMPARISON_TOLERANCE_EV:.1e} eV."
    )


def write_audit(
    *,
    config: Config,
    data: pd.DataFrame,
    target_col: str,
    input_records: Sequence[Mapping[str, object]],
    assertions: Sequence[AssertionResult],
    summary: pd.DataFrame,
    spread_table: pd.DataFrame,
    size_distribution: pd.DataFrame,
    bootstrap: pd.DataFrame,
    spread_bootstrap: pd.DataFrame,
    constancy_stats: Mapping[str, object],
    constancy_exception_path: Optional[Path],
    output_files: Sequence[Path],
) -> None:
    legacy = summary.loc[
        (summary["analysis"] == "legacy_reproduction") & (summary["fold"] == "pooled")
    ].iloc[0]
    a_overall = summary.loc[
        (summary["analysis"] == "fold_conditioned")
        & (summary["population"] == POPULATION_A)
        & (summary["fold"] == "pooled")
    ].iloc[0]
    b_overall = summary.loc[
        (summary["analysis"] == "fold_conditioned")
        & (summary["population"] == POPULATION_B)
        & (summary["fold"] == "pooled")
    ].iloc[0]

    input_rows = []
    for record in input_records:
        input_rows.append(
            {
                "level": record["level"],
                "fold": record["fold"],
                "path": record["path"],
                "mapping": record["mapping"],
                "sha256": str(record["sha256"])[:16] + "…",
            }
        )

    assertion_rows = [
        {
            "assertion": result.name,
            "status": "PASS" if result.passed else "FAIL",
            "detail": result.detail,
        }
        for result in assertions
    ]

    overall_rows = []
    for row in (legacy, a_overall, b_overall):
        overall_rows.append(
            {
                "analysis": row["analysis"],
                "population": row["population"],
                "n_samples": f"{int(row['n_samples']):,}",
                "n_formulas": f"{int(row['n_formulas']):,}",
                "n_formula_fold_groups": f"{int(row['n_formula_fold_groups']):,}",
                "bound": format_float(float(row["oracle_bound_eV"])),
                "L1": format_float(float(row["L1_MAE_eV"])),
                "L2": format_float(float(row["L2_MAE_eV"])),
                "L3": format_float(float(row["L3_MAE_eV"])),
            }
        )

    old_spread = spread_table.loc[spread_table["analysis"] == "legacy_reproduction"]
    old_spread_rows = [
        {
            "bin": row["spread_bin"],
            "n_samples": f"{int(row['n_samples']):,}",
            "n_formulas": f"{int(row['n_formulas']):,}",
            "bound": format_float(float(row["oracle_bound_eV"])),
        }
        for _, row in old_spread.iterrows()
    ]

    fold_rows = summary.loc[
        (summary["analysis"] == "fold_conditioned")
        & (summary["population"] == POPULATION_A)
        & (summary["fold"] != "pooled")
    ]
    per_fold_rows = [
        {
            "fold": int(row["fold"]),
            "n_samples": f"{int(row['n_samples']):,}",
            "n_groups": f"{int(row['n_formula_fold_groups']):,}",
            "bound": format_float(float(row["oracle_bound_eV"])),
        }
        for _, row in fold_rows.iterrows()
    ]

    ci_rows = []
    for _, row in bootstrap.iterrows():
        ci_rows.append(
            {
                "population": row["population"],
                "metric": row["metric"],
                "estimate": format_float(float(row["point_estimate"])),
                "lower": format_float(float(row["ci_95_lower"])),
                "upper": format_float(float(row["ci_95_upper"])),
            }
        )

    same_fold_spread_ci_rows = []
    same_fold_spread_bootstrap = spread_bootstrap.loc[
        spread_bootstrap["population"] == POPULATION_B
    ]
    for _, row in same_fold_spread_bootstrap.iterrows():
        same_fold_spread_ci_rows.append(
            {
                "bin": row["spread_bin"],
                "bound": format_float(float(row["oracle_bound_eV"])),
                "L2_delta": format_float(float(row["L2_MAE_minus_bound_eV"])),
                "L2_CI": (
                    f"[{float(row['L2_ci_95_lower']):.6f}, "
                    f"{float(row['L2_ci_95_upper']):.6f}]"
                ),
                "L2_decision": row["L2_ci_zero_classification"],
                "L3_delta": format_float(float(row["L3_MAE_minus_bound_eV"])),
                "L3_CI": (
                    f"[{float(row['L3_ci_95_lower']):.6f}, "
                    f"{float(row['L3_ci_95_upper']):.6f}]"
                ),
                "L3_decision": row["L3_ci_zero_classification"],
            }
        )

    lines = [
        "# Fold-conditioned repeated-composition oracle audit",
        "",
        f"- Run time: `{datetime.now().astimezone().isoformat()}`",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{np.__version__}`",
        f"- pandas: `{pd.__version__}`",
        f"- MatBench: `{package_version('matbench')}`",
        f"- Task: `{config.task_name}`",
        f"- Target column: `{target_col}`",
        f"- Official OOF samples: `{len(data):,}`",
        f"- Official folds: `{list(config.folds)}`",
        "",
        "## Inputs and sample connection",
        "",
        "L1/L2 NPY files contain predictions only. Their IDs were reconstructed from the "
        "official test-fold order established by the reviewed generation pipelines. "
        "L3 was joined using IDs stored in each NPZ. After ingestion, every merge and "
        "subset operation used stable `mbid`, never the current DataFrame row order.",
        "",
        markdown_table(
            input_rows,
            (
                ("level", "Level"),
                ("fold", "Fold"),
                ("path", "Input file"),
                ("mapping", "ID mapping"),
                ("sha256", "SHA-256 prefix"),
            ),
        ),
        "",
        "## Definitions",
        "",
        "- `global fixed-function oracle bound`: formula-level median absolute deviation "
        "for one fixed pooled function f(c). It is not a strict lower bound for pooled "
        "OOF predictions from fold-specific models.",
        "- `fold-conditioned OOF oracle bound`: median absolute deviation within each "
        "`(reduced_formula, official_test_fold)` group.",
        f"- `{POPULATION_A}`: all entries whose formula occurs at least twice globally.",
        f"- `{POPULATION_B}`: only complete `(formula, fold)` groups of size at least two.",
        "- Group and population bounds are sample-weighted.",
        "- Formula names alone are described as repeated-composition entries/same-formula "
        "groups; no claim of structurally distinct polymorphism is made.",
        "",
        "## Legacy reproduction",
        "",
        f"- Repeated-composition entries: `{int(legacy['n_samples']):,}`",
        f"- Same-formula groups: `{int(legacy['n_formulas']):,}`",
        f"- Global fixed-function oracle bound: `{float(legacy['oracle_bound_eV']):.8f} eV` "
        f"(`{float(legacy['oracle_bound_eV']):.4f} eV` at four decimals)",
        "",
        markdown_table(
            old_spread_rows,
            (
                ("bin", "Global formula spread bin"),
                ("n_samples", "n samples"),
                ("n_formulas", "n formulas"),
                ("bound", "Old global bound (eV)"),
            ),
        ),
        "",
        "The old bound is retained only under the name **global fixed-function oracle "
        "bound**. It must not be presented as a strict pooled-OOF composition-only bound.",
        "",
        "## Main populations",
        "",
        markdown_table(
            overall_rows,
            (
                ("analysis", "Analysis"),
                ("population", "Population"),
                ("n_samples", "n samples"),
                ("n_formulas", "n formulas"),
                ("n_formula_fold_groups", "n (formula, fold) groups"),
                ("bound", "Oracle bound (eV)"),
                ("L1", "L1 MAE"),
                ("L2", "L2 MAE"),
                ("L3", "L3 MAE"),
            ),
        ),
        "",
        "For the global repeated-formula population:",
        "",
        f"- size=1 groups: `{int(a_overall['n_size1_groups']):,}` groups / "
        f"`{int(a_overall['n_size1_samples']):,}` samples; their oracle contribution is zero.",
        f"- size>=2 groups: `{int(a_overall['n_size_ge2_groups']):,}` groups / "
        f"`{int(a_overall['n_size_ge2_samples']):,}` samples.",
        "",
        "Per-fold fold-conditioned bounds for the global repeated-formula population:",
        "",
        markdown_table(
            per_fold_rows,
            (
                ("fold", "Fold"),
                ("n_samples", "n samples"),
                ("n_groups", "n groups"),
                ("bound", "Bound (eV)"),
            ),
        ),
        "",
        "The complete exact group-size distribution is in "
        "`polymorph_bound_group_size_distribution.csv`.",
        "",
        "## Level-1 within-group constancy",
        "",
        f"- Groups checked (size>=2): `{int(constancy_stats['n_groups_checked']):,}`",
        f"- Maximum within-group range: `{float(constancy_stats['max_range_eV']):.12g} eV`",
        f"- Maximum within-group population standard deviation: "
        f"`{float(constancy_stats['max_std_eV']):.12g} eV`",
        f"- Strictly nonzero groups: `{int(constancy_stats['n_strictly_nonzero_groups']):,}`",
        f"- Groups exceeding tolerance: "
        f"`{int(constancy_stats['n_groups_exceeding_tolerance']):,}`",
        f"- Numerical tolerance: `{float(constancy_stats['tolerance_eV']):.12g} eV`",
    ]
    if constancy_exception_path is not None:
        lines.extend(
            [
                f"- Every nonzero exception is listed in `{constancy_exception_path.name}`.",
            ]
        )
    else:
        lines.extend(["- No nonzero Level-1 within-group ranges were found."])

    lines.extend(
        [
            "",
            "## Formula-cluster bootstrap",
            "",
        ]
    )
    if len(bootstrap):
        lines.extend(
            [
                f"- Bootstrap replicates: `{config.bootstrap_replicates:,}`",
                f"- Random seed: `{config.bootstrap_seed}`",
                "- Interval method: percentile bootstrap (2.5th and 97.5th percentiles).",
                "- Cluster unit: `reduced_formula`; every formula's entries across all folds "
                "remain together in a resample.",
                "",
                markdown_table(
                    ci_rows,
                    (
                        ("population", "Population"),
                        ("metric", "Metric"),
                        ("estimate", "Estimate"),
                        ("lower", "95% lower"),
                        ("upper", "95% upper"),
                    ),
                ),
            ]
        )
    else:
        lines.extend(
            [
                "- **Not completed:** bootstrap was disabled for this run. No confidence "
                "intervals are claimed.",
            ]
        )

    lines.extend(
        [
            "",
            "## Global-spread-bin formula-cluster bootstrap",
            "",
        ]
    )
    if len(spread_bootstrap):
        lines.extend(
            [
                "- Spread definition: `global within-formula label spread`.",
                f"- Bootstrap replicates per population/bin: `{config.bootstrap_replicates:,}`.",
                f"- Random seed: `{config.bootstrap_seed}`.",
                "- Interval method: percentile bootstrap (2.5th and 97.5th percentiles).",
                "- Cluster unit: `reduced_formula`; all folds and samples of a formula "
                "within the named analysis population remain together.",
                "- Resampling multiplicity is preserved by repeated integer-array indexing; "
                "there is no post-draw groupby, unique operation, or deduplication.",
                "- Decision rule: CI upper < 0 = robust undercut; CI lower > 0 = robust "
                "above bound; otherwise the CI includes zero and only the point estimate "
                "is reported as directional evidence.",
                "",
                f"Main table for `{POPULATION_B}`:",
                "",
                markdown_table(
                    same_fold_spread_ci_rows,
                    (
                        ("bin", "Global spread bin"),
                        ("bound", "Oracle bound"),
                        ("L2_delta", "L2 - bound"),
                        ("L2_CI", "L2 95% CI"),
                        ("L2_decision", "L2 decision"),
                        ("L3_delta", "L3 - bound"),
                        ("L3_CI", "L3 95% CI"),
                        ("L3_decision", "L3 decision"),
                    ),
                ),
                "",
                "Both populations are available in "
                "`polymorph_bound_bootstrap_by_spread.csv`.",
                "The main same-fold CI evidence is visualized in "
                "`polymorph_bound_delta_ci_by_spread.png`.",
            ]
        )
    else:
        lines.extend(
            [
                "- **Not completed:** spread-bin bootstrap was disabled for this run. "
                "No bin-level confidence intervals are claimed.",
            ]
        )

    lines.extend(
        [
            "",
            "## Assertions",
            "",
            markdown_table(
                assertion_rows,
                (
                    ("assertion", "Assertion"),
                    ("status", "Status"),
                    ("detail", "Detail"),
                ),
            ),
            "",
            "## Interpretation",
            "",
            f"### {POPULATION_A}",
            "",
            f"- {relation_sentence('L2', a_overall)}",
            f"- {relation_sentence('L3', a_overall)}",
            "",
            f"### {POPULATION_B}",
            "",
            f"- {relation_sentence('L2', b_overall)}",
            f"- {relation_sentence('L3', b_overall)}",
            "",
            "The manuscript's old `0.1539 eV` value may be retained only with the renamed "
            "global fixed-function interpretation above. The old “32% undercut” statement "
            "is not carried forward automatically: it must be withdrawn or replaced by the "
            "computed fold-conditioned percentage in the corresponding population/spread row. "
            "An undercut percentage is populated in the CSV only when model MAE is truly below "
            "the relevant bound; otherwise the separate above-bound percentage is reported.",
            "",
            "## Output files",
            "",
        ]
    )
    lines.extend(f"- `{path.name}`" for path in output_files)
    lines.append("")

    audit_path = config.output_dir / "polymorph_bound_audit.md"
    audit_path.write_text("\n".join(lines), encoding="utf-8")


def print_key_results(summary: pd.DataFrame, spread_table: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("KEY RESULTS")
    print("=" * 88)
    overall = summary.loc[summary["fold"] == "pooled"]
    columns = [
        "analysis",
        "population",
        "n_samples",
        "n_formulas",
        "n_formula_fold_groups",
        "oracle_bound_eV",
        "L1_MAE_eV",
        "L2_MAE_eV",
        "L3_MAE_eV",
    ]
    print(overall[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\nFold-conditioned per-fold bounds (population A):")
    per_fold = summary.loc[
        (summary["analysis"] == "fold_conditioned")
        & (summary["population"] == POPULATION_A)
        & (summary["fold"] != "pooled")
    ]
    print(
        per_fold[["fold", "n_samples", "n_formula_fold_groups", "oracle_bound_eV"]]
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )

    print("\nLegacy global-spread-bin reproduction:")
    legacy_spread = spread_table.loc[spread_table["analysis"] == "legacy_reproduction"]
    print(
        legacy_spread[["spread_bin", "n_samples", "n_formulas", "oracle_bound_eV"]]
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


def run_self_test() -> None:
    assertions: List[AssertionResult] = []
    synthetic = pd.DataFrame(
        {
            "mbid": ["a0", "a1", "a2", "b0", "b1"],
            "test_fold": [0, 0, 1, 0, 1],
            "fold_position": [0, 1, 0, 2, 1],
            "reduced_formula": ["A", "A", "A", "B", "B"],
            "y": [0.0, 2.0, 10.0, 5.0, 5.0],
            "pred_L1": [1.0, 1.0, 10.0, 5.0, 5.0],
            "pred_L2": [0.5, 1.5, 9.0, 5.0, 5.0],
            "pred_L3": [0.0, 2.0, 10.0, 5.0, 5.0],
        }
    )
    synthetic = add_global_formula_columns(synthetic)
    for level in ("L1", "L2", "L3"):
        synthetic[f"ae_{level}"] = (synthetic["y"] - synthetic[f"pred_{level}"]).abs()

    a, b = derive_fold_conditioned_populations(synthetic, assertions)
    assert math.isclose(float(a["global_fixed_bound_ae"].mean()), 2.0)
    assert math.isclose(float(a["fold_conditioned_bound_ae"].mean()), 0.4)
    assert len(b) == 2
    assert b["reduced_formula"].nunique() == 1

    exceptions, stats = l1_constancy_table(a, 1e-12, assertions)
    assert exceptions.empty
    assert stats["max_range_eV"] == 0.0

    summary = build_summary(a, b, (0, 1))
    assert len(summary) == 7
    a_summary = summary.loc[
        (summary["analysis"] == "fold_conditioned")
        & (summary["population"] == POPULATION_A)
        & (summary["fold"] == "pooled")
    ].iloc[0]
    b_summary = summary.loc[
        (summary["analysis"] == "fold_conditioned")
        & (summary["population"] == POPULATION_B)
        & (summary["fold"] == "pooled")
    ].iloc[0]
    assert int(a_summary["n_size1_groups"]) == 3
    assert int(a_summary["n_size1_samples"]) == 3
    assert int(a_summary["n_size_ge2_groups"]) == 1
    assert int(a_summary["n_size_ge2_samples"]) == 2
    assert a_summary["L2_relation"] == "equal_within_tolerance"
    assert b_summary["L2_relation"] == "undercut"
    assert math.isclose(float(b_summary["L2_undercut_percent"]), 50.0)
    spread = build_spread_table(a, b)
    assert len(spread) == 12
    sizes = build_group_size_distribution(a, b)
    assert int(sizes["n_groups"].sum()) == 5
    bootstrap = cluster_bootstrap(
        a,
        population=POPULATION_A,
        n_replicates=25,
        seed=123,
    )
    assert len(bootstrap) == 3
    assert bool(np.isfinite(bootstrap["ci_95_lower"]).all())

    # A repeated index must contribute repeatedly; no unique/groupby step may
    # collapse formula multiplicity after the draw.
    numerator = np.array([1.0, 9.0])
    denominator = np.array([1.0, 1.0])
    repeated_draw = np.array([0, 0, 1])
    assert math.isclose(
        resampled_ratio(numerator, denominator, repeated_draw), 11.0 / 3.0
    )
    assert not math.isclose(
        resampled_ratio(numerator, denominator, repeated_draw),
        resampled_ratio(numerator, denominator, np.unique(repeated_draw)),
    )

    bin_rows = []
    for bin_index, spread_label in enumerate(SPREAD_LABELS):
        formula = f"BIN{bin_index}"
        for fold in (0, 1):
            for within_fold in (0, 1):
                bound_ae = 0.2 + 0.01 * bin_index
                bin_rows.append(
                    {
                        "mbid": f"{formula}_{fold}_{within_fold}",
                        "test_fold": fold,
                        "reduced_formula": formula,
                        "spread_bin": spread_label,
                        "fold_conditioned_bound_ae": bound_ae,
                        "ae_L2": bound_ae + 0.05,
                        "ae_L3": bound_ae - 0.03,
                    }
                )
    bin_frame = pd.DataFrame(bin_rows)
    spread_bootstrap = build_spread_bootstrap_table(
        bin_frame, bin_frame, n_replicates=25, seed=123
    )
    assert len(spread_bootstrap) == 8
    assert bool(spread_bootstrap["multiplicity_preserved"].all())
    assert bool(
        (
            spread_bootstrap["L2_ci_zero_classification"]
            == "robust_above_bound_ci_lower_above_zero"
        ).all()
    )
    assert bool(
        (
            spread_bootstrap["L3_ci_zero_classification"]
            == "robust_undercut_ci_upper_below_zero"
        ).all()
    )
    print(
        "SELF-TEST PASSED: grouping, oracle formulas, subsets, spread bins, "
        "formula-cluster multiplicity, pooled bootstrap, and bin-level CI rules."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    assertions: List[AssertionResult] = []
    input_records: List[Dict[str, object]] = []

    data, official_ids, target_col = load_official_oof_data(config, assertions)
    data = add_global_formula_columns(data)

    # Hard gate: do not compute new conclusions unless the old population and
    # global fixed-function bound reproduce first.
    population_a_base = legacy_reproduction_gate(data, config.output_dir, assertions)

    prediction_paths = collect_prediction_paths(config)
    require_all_prediction_files(prediction_paths)
    add_assertion(
        assertions,
        "all 15 frozen prediction files exist",
        True,
        "5 folds each for L1, L2, and L3",
    )
    print("\nLoading frozen clipped predictions...")
    l1 = load_position_bound_predictions(
        "L1", prediction_paths["L1"], official_ids, input_records, assertions
    )
    l2 = load_position_bound_predictions(
        "L2", prediction_paths["L2"], official_ids, input_records, assertions
    )
    l3 = load_id_keyed_l3_predictions(
        config, prediction_paths["L3"], official_ids, input_records, assertions
    )
    data = attach_predictions_by_id(data, (l1, l2, l3), assertions)
    for level in ("L1", "L2", "L3"):
        values = data[f"pred_{level}"].to_numpy(dtype=float)
        add_assertion(
            assertions,
            f"{level} predictions are finite and respect nonnegative clipping",
            bool(np.isfinite(values).all())
            and float(values.min()) >= -COMPARISON_TOLERANCE_EV,
            (
                f"n={len(values)}; n_nonfinite={int((~np.isfinite(values)).sum())}; "
                f"minimum={float(values.min()):.12g}"
            ),
        )

    # Rebuild A from the ID-joined table so labels and all prediction columns are
    # guaranteed to cover the identical sample set.
    population_a_base = data.loc[data["formula_global_size"] >= 2].copy()
    population_a, population_b = derive_fold_conditioned_populations(
        population_a_base, assertions
    )

    constancy_exceptions, constancy_stats = l1_constancy_table(
        population_a, config.l1_tolerance, assertions
    )
    constancy_exception_path: Optional[Path] = None
    if len(constancy_exceptions):
        constancy_exception_path = (
            config.output_dir / "polymorph_bound_l1_constancy_exceptions.csv"
        )
        constancy_exceptions.to_csv(constancy_exception_path, index=False)

    # When L1 is constant, its error cannot be below the within-group median
    # oracle. Record this as an additional end-to-end alignment check.
    l1_constant_at_tolerance = (
        int(constancy_stats["n_groups_exceeding_tolerance"]) == 0
    )
    for population_name, frame in (
        (POPULATION_A, population_a),
        (POPULATION_B, population_b),
    ):
        bound = float(frame["fold_conditioned_bound_ae"].mean())
        l1_mae = float(frame["ae_L1"].mean())
        add_assertion(
            assertions,
            f"L1 MAE is not below the fold-conditioned oracle in {population_name}",
            (not l1_constant_at_tolerance)
            or l1_mae + config.l1_tolerance >= bound,
            f"L1_MAE={l1_mae:.12g}; bound={bound:.12g}",
        )

    summary = build_summary(population_a, population_b, config.folds)
    spread_table = build_spread_table(population_a, population_b)
    size_distribution = build_group_size_distribution(population_a, population_b)

    bootstrap_frames = []
    if config.bootstrap_replicates > 0:
        for population_name, frame in (
            (POPULATION_A, population_a),
            (POPULATION_B, population_b),
        ):
            bootstrap_frames.append(
                cluster_bootstrap(
                    frame,
                    population=population_name,
                    n_replicates=config.bootstrap_replicates,
                    seed=config.bootstrap_seed,
                )
            )
        bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
        add_assertion(
            assertions,
            "formula-cluster bootstrap completed for both populations",
            set(bootstrap["population"]) == {POPULATION_A, POPULATION_B}
            and len(bootstrap) == 6,
            f"rows={len(bootstrap)}; replicates={config.bootstrap_replicates}; seed={config.bootstrap_seed}",
        )
    else:
        bootstrap = pd.DataFrame(
            columns=[
                "population",
                "metric",
                "point_estimate",
                "ci_95_lower",
                "ci_95_upper",
                "n_bootstrap",
                "seed",
                "cluster_unit",
                "n_unique_formula_clusters",
            ]
        )

    spread_bootstrap = build_spread_bootstrap_table(
        population_a,
        population_b,
        n_replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    add_assertion(
        assertions,
        "global-spread-bin formula-cluster bootstrap completed for both populations",
        len(spread_bootstrap) == 2 * len(SPREAD_LABELS)
        and set(spread_bootstrap["population"]) == {POPULATION_A, POPULATION_B},
        (
            f"rows={len(spread_bootstrap)}; expected={2 * len(SPREAD_LABELS)}; "
            f"replicates={config.bootstrap_replicates}; seed={config.bootstrap_seed}"
        ),
    )
    validate_spread_bootstrap_table(
        spread_bootstrap,
        spread_table,
        config=config,
        assertions=assertions,
    )

    # Full ID-keyed manifest. Formula/fold group sizes are computed on the full
    # official dataset; every same-fold repeated group is necessarily part of A.
    manifest = data.copy()
    manifest["formula_fold_size"] = manifest.groupby(
        ["reduced_formula", "test_fold"], sort=False
    )["mbid"].transform("size").astype(int)
    manifest["is_global_repeated_formula"] = manifest["formula_global_size"] >= 2
    manifest["is_same_fold_repeated"] = manifest["formula_fold_size"] >= 2
    manifest_columns = [
        "mbid",
        "test_fold",
        "fold_position",
        "reduced_formula",
        "y",
        "pred_L1",
        "pred_L2",
        "pred_L3",
        "formula_global_size",
        "formula_fold_size",
        "global_formula_spread_eV",
        "is_global_repeated_formula",
        "is_same_fold_repeated",
    ]

    manifest_path = config.output_dir / "polymorph_bound_prediction_manifest.csv"
    summary_path = config.output_dir / "polymorph_bound_summary.csv"
    spread_path = config.output_dir / "polymorph_bound_by_spread.csv"
    size_path = config.output_dir / "polymorph_bound_group_size_distribution.csv"
    bootstrap_path = config.output_dir / "polymorph_bound_bootstrap.csv"
    spread_bootstrap_path = (
        config.output_dir / "polymorph_bound_bootstrap_by_spread.csv"
    )
    top_path = config.output_dir / "repeated_composition_groups_top.csv"
    plot_path = config.output_dir / "polymorph_bound_by_spread.png"
    ci_plot_path = config.output_dir / "polymorph_bound_delta_ci_by_spread.png"
    audit_path = config.output_dir / "polymorph_bound_audit.md"

    manifest[manifest_columns].to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)
    spread_table.to_csv(spread_path, index=False)
    size_distribution.to_csv(size_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    spread_bootstrap.to_csv(spread_bootstrap_path, index=False)

    top_formulas = (
        population_a.groupby("reduced_formula", sort=False)["global_formula_spread_eV"]
        .first()
        .sort_values(ascending=False)
        .head(30)
        .index
    )
    top_columns = [
        "reduced_formula",
        "mbid",
        "test_fold",
        "y",
        "global_formula_spread_eV",
        "formula_fold_size",
        "global_fixed_bound_ae",
        "fold_conditioned_bound_ae",
        "pred_L1",
        "pred_L2",
        "pred_L3",
    ]
    (
        population_a.loc[population_a["reduced_formula"].isin(top_formulas), top_columns]
        .sort_values(
            ["global_formula_spread_eV", "reduced_formula", "test_fold", "y"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        .to_csv(top_path, index=False)
    )

    output_files = [
        manifest_path,
        summary_path,
        spread_path,
        size_path,
        bootstrap_path,
        spread_bootstrap_path,
        top_path,
    ]
    if constancy_exception_path is not None:
        output_files.append(constancy_exception_path)
    if config.make_plot:
        make_spread_plot(spread_table, plot_path)
        make_spread_ci_plot(spread_bootstrap, ci_plot_path)
        output_files.append(plot_path)
        output_files.append(ci_plot_path)
    output_files.append(audit_path)

    print_key_results(summary, spread_table)
    print(f"\nSpread-bin bootstrap ({POPULATION_B}):")
    print(
        spread_bootstrap.loc[
            spread_bootstrap["population"] == POPULATION_B,
            [
                "spread_bin",
                "oracle_bound_eV",
                "L2_MAE_minus_bound_eV",
                "L2_ci_95_lower",
                "L2_ci_95_upper",
                "L2_ci_zero_classification",
                "L3_MAE_minus_bound_eV",
                "L3_ci_95_lower",
                "L3_ci_95_upper",
                "L3_ci_zero_classification",
            ],
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    write_audit(
        config=config,
        data=manifest,
        target_col=target_col,
        input_records=input_records,
        assertions=assertions,
        summary=summary,
        spread_table=spread_table,
        size_distribution=size_distribution,
        bootstrap=bootstrap,
        spread_bootstrap=spread_bootstrap,
        constancy_stats=constancy_stats,
        constancy_exception_path=constancy_exception_path,
        output_files=output_files,
    )

    print("\nOutputs written to:")
    for path in output_files:
        print(f"  {path}")

    failed = [result for result in assertions if not result.passed]
    if failed:
        names = "; ".join(result.name for result in failed)
        raise AssertionError(
            "Analysis outputs were written, but one or more audit assertions failed. "
            f"Do not use the conclusions until resolved: {names}"
        )
    print("\nAll audit assertions passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
