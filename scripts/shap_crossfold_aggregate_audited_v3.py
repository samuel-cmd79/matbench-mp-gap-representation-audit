"""
SHAP 跨 fold 聚合 — audited v3（完整 L1/L2 + L1→L2 transfer）
================================================================

这是在原 shap_crossfold_aggregate_audited.py 上继续修改的版本。L1 和 L2
分别输出完整的全特征表、组汇总、逐 fold 组占比、Top-20 图和组贡献图；
文件名前缀严格分为 v1_audited_v3_* 与 v2_audited_v3_*。组占比口径是每个 fold
内先归一化，再对五个 fold 等权平均。L1→L2 transfer 单独使用未归一化
的绝对 mean(|SHAP|)，输出为 v1_to_v2_transfer_v3_*。

新增修正：
1. 精确列名 ``band center`` 优先独立归入 ``BandCenter (L2 only)``；
2. Ionicity 只包含六个 Electronegativity 特征；
3. 从完整 L2、symmetry deletion、coordination deletion 的 scores_xgb.txt
   读取 mean MAE，生成规范 deletion CSV，并接入最终五列表；
4. 只对 L1/L2 共有的 composition features 比较绝对 mean(|SHAP|)，
   明确排除 BandCenter，并报告 L1 分母；
5. 检查 L2 共 283 列全部互斥、无重复且无遗漏。

v3 使用全新的输出目录和前缀，不会覆盖旧脚本的结果。正式输出合同固定为：

- 10 个 ``v1_audited_v3_*`` 文件；
- 12 个 ``v2_audited_v3_*`` 文件；
- 3 个 ``v1_to_v2_transfer_v3_*`` 文件。

输出目录只允许存在上述 25 个文件；发现其他非隐藏文件时脚本会停止，
从而避免不同版本的历史结果再次混在一起。

默认目录与已冻结结果一致，通常直接运行即可：

    python shap_crossfold_aggregate_audited_v3.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ================= CONFIG =================
DEFAULT_L1_SHAP_DIR = Path("../outputs_v1_run0709")
DEFAULT_L2_SHAP_DIR = Path("../matbench_outputs_v2_run0709")
DEFAULT_FULL_L2_SCORE = Path(
    "../matbench_outputs_v2_run0709/scores_xgb.txt"
)
DEFAULT_SYMMETRY_SCORE = Path(
    "../matbench_outputs_v2_ablation_symmetry/scores_xgb.txt"
)
DEFAULT_COORDINATION_SCORE = Path(
    "../matbench_outputs_v2_ablation_coordination/scores_xgb.txt"
)
DEFAULT_OUTPUT_DIR = Path("./shap_aggregate_audited_v3")
DEFAULT_MODEL = "xgb"
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
DEFAULT_V1_PREFIX = "v1_audited_v3"
DEFAULT_V2_PREFIX = "v2_audited_v3"
DEFAULT_TRANSFER_PREFIX = "v1_to_v2_transfer_v3"
DEFAULT_L1_TITLE_TAG = "Level 1 (composition, XGB)"
DEFAULT_TITLE_TAG = "Level 2 (+descriptors, XGB)"
TOPK_PLOT = 20
EXPECTED_L2_COLUMNS = 283

# 唯一采用的正式聚合口径：
AGGREGATION_METHOD = "normalize within each fold, then average the five folds equally"

# BandCenter 是独立 L2 组，明确不计入结构总占比。
INCLUDE_BAND_CENTER_IN_STRUCTURAL_TOTAL = False

# structural total 是子总计，不是额外的互斥组。
STRUCTURAL_GROUPS = (
    "Global symmetry",
    "Coordination fingerprint",
    "Bond length / packing",
    "Chemical ordering",
    "Dimensionality",
    "Other structure",
)

MAJOR_STRUCTURAL_GROUPS = (
    "Global symmetry",
    "Coordination fingerprint",
    "Bond length / packing",
    "Chemical ordering",
)

COMPOSITION_GROUPS = (
    "Bond strength (comp)",
    "Ionicity",
    "d/p electron count (comp)",
    "Valence total (comp)",
    "Band-gap prior (comp)",
    "Atomic size (comp)",
    "Periodic position (comp)",
    "Unfilled orbitals (comp)",
    "Magnetic moment (comp)",
    "Other composition",
)

# 新要求给出的六位小数核验锚点。
REFERENCE_DECIMALS = 6
REFERENCE_ABS_TOLERANCE_PCT = 1e-5
REFERENCE_BANDCENTER_COLUMNS = 1
REFERENCE_BANDCENTER_SHARE = 0.642706
REFERENCE_IONICITY_COLUMNS = 6
REFERENCE_IONICITY_SHARE = 6.286173
REFERENCE_IONICITY_DENSITY = 1.047696
REFERENCE_STRUCTURAL_TOTAL = 46.505920
REFERENCE_MAJOR_STRUCTURAL_ROUNDED = 46.2
REFERENCE_SECONDARY_STRUCTURAL_ROUNDED = 0.3
REFERENCE_IONICITY_TRANSFER_ROUNDED = -20.1

# 重复列名或同时命中多个具体分组时停止，不静默“先到先得”。
FAIL_ON_DUPLICATE_OR_OVERLAP = True
DEFAULT_FAIL_ON_FALLBACK = False
# ==========================================


Rule = tuple[str, Callable[[str], bool]]


def is_band_center(name: str) -> bool:
    # 新要求指定精确名称；大小写或额外空格不静默接受。
    return name == "band center"


# 这里只放具体规则。Other composition / Other structure 在没有具体命中时
# 作为 fallback 使用，从而避免原脚本中宽泛规则制造表面重叠。
SPECIFIC_GROUP_RULES: list[Rule] = [
    (
        "Global symmetry",
        lambda n: n in ("n_symmetry_ops", "spacegroup_num")
        or n.startswith("crystal_system")
        or n == "is_centrosymmetric",
    ),
    (
        "Coordination fingerprint",
        lambda n: (" CN_" in n) or n.startswith("mean wt CN"),
    ),
    (
        "Bond length / packing",
        lambda n: "bond length" in n
        or "neighbor distance" in n
        or n in ("density", "vpa", "packing fraction"),
    ),
    ("Chemical ordering", lambda n: "ordering parameter" in n),
    ("Dimensionality", lambda n: n.startswith("dimensionality")),
    ("BandCenter (L2 only)", is_band_center),
    ("Bond strength (comp)", lambda n: "MeltingT" in n),
    ("Ionicity", lambda n: "Electronegativity" in n),
    (
        "d/p electron count (comp)",
        lambda n: "NdValence" in n
        or "NpValence" in n
        or "NfValence" in n
        or "NsValence" in n,
    ),
    ("Valence total (comp)", lambda n: "NValence" in n),
    ("Band-gap prior (comp)", lambda n: "GSbandgap" in n),
    (
        "Atomic size (comp)",
        lambda n: "CovalentRadius" in n
        or "GSvolume" in n
        or "AtomicWeight" in n
        or "Row" in n,
    ),
    (
        "Periodic position (comp)",
        lambda n: "Column" in n
        or "MendeleevNumber" in n
        or "Number" in n,
    ),
    ("Unfilled orbitals (comp)", lambda n: "Unfilled" in n),
    ("Magnetic moment (comp)", lambda n: "GSmagmom" in n),
]


GROUP_COLORS = {
    group: color
    for group, color in zip(
        [
            "Global symmetry",
            "Coordination fingerprint",
            "Bond length / packing",
            "Chemical ordering",
            "Dimensionality",
            "BandCenter (L2 only)",
            "Bond strength (comp)",
            "Ionicity",
            "d/p electron count (comp)",
            "Valence total (comp)",
            "Band-gap prior (comp)",
            "Atomic size (comp)",
            "Periodic position (comp)",
            "Unfilled orbitals (comp)",
            "Magnetic moment (comp)",
            "Other composition",
            "Other structure",
        ],
        plt.cm.tab20.colors,
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete corrected L1/L2 SHAP outputs and transfer audit."
    )
    parser.add_argument("--l1-shap-dir", type=Path, default=DEFAULT_L1_SHAP_DIR)
    parser.add_argument("--l2-shap-dir", type=Path, default=DEFAULT_L2_SHAP_DIR)
    parser.add_argument("--full-l2-score", type=Path, default=DEFAULT_FULL_L2_SCORE)
    parser.add_argument(
        "--symmetry-score", type=Path, default=DEFAULT_SYMMETRY_SCORE
    )
    parser.add_argument(
        "--coordination-score", type=Path, default=DEFAULT_COORDINATION_SCORE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--folds",
        default=",".join(map(str, DEFAULT_FOLDS)),
        help="Comma-separated fold ids, e.g. 0,1,2,3,4",
    )
    parser.add_argument("--v1-prefix", default=DEFAULT_V1_PREFIX)
    parser.add_argument("--v2-prefix", default=DEFAULT_V2_PREFIX)
    parser.add_argument("--transfer-prefix", default=DEFAULT_TRANSFER_PREFIX)
    parser.add_argument("--title-tag", default=DEFAULT_TITLE_TAG)
    parser.add_argument(
        "--fail-on-fallback",
        action="store_true",
        default=DEFAULT_FAIL_ON_FALLBACK,
        help="Stop if any feature is assigned through a generic fallback.",
    )
    parser.add_argument(
        "--fail-on-reference-mismatch",
        action="store_true",
        help="Stop unless L2 reference anchors match within the stated tolerance.",
    )
    return parser.parse_args()


def parse_folds(text: str) -> list[int]:
    try:
        folds = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --folds value: {text!r}") from exc
    if not folds or len(folds) != len(set(folds)):
        raise ValueError("--folds must contain at least one unique integer fold id.")
    return folds


def expected_output_names(args: argparse.Namespace) -> set[str]:
    """Return the exact 25-file v3 output contract."""
    level_suffixes = {
        "aggregation_audit_report.txt",
        "feature_group_audit.csv",
        "group_share_by_fold.csv",
        "paper_group_summary.csv",
        "reference_share_check.csv",
        "shap_crossfold_full.csv",
        f"shap_crossfold_top{TOPK_PLOT}.png",
        "shap_group_contribution.csv",
        "shap_group_contribution.png",
        "structural_total_check.csv",
    }
    expected = {
        f"{prefix}_{suffix}"
        for prefix in (args.v1_prefix, args.v2_prefix)
        for suffix in level_suffixes
    }
    expected.update(
        {
            f"{args.v2_prefix}_deletion_ablation.csv",
            f"{args.v2_prefix}_manuscript_sentence.txt",
            f"{args.transfer_prefix}_features.csv",
            f"{args.transfer_prefix}_groups.csv",
            f"{args.transfer_prefix}_metadata.json",
        }
    )
    if len(expected) != 25:
        raise AssertionError(
            "Output prefixes collide; v3 requires 25 unique output names."
        )
    return expected


def guard_output_directory(output_dir: Path, expected: set[str]) -> None:
    """Refuse to mix v3 outputs with files from another script version."""
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in output_dir.iterdir()
        if not path.name.startswith(".")
        and (not path.is_file() or path.name not in expected)
    )
    if unexpected:
        preview = ", ".join(unexpected[:10])
        more = " ..." if len(unexpected) > 10 else ""
        raise ValueError(
            f"Output directory {output_dir} contains files outside the v3 "
            f"25-file contract: {preview}{more}. Use the new empty v3 "
            "directory or another --output-dir."
        )


def verify_output_contract(output_dir: Path, expected: set[str]) -> None:
    """Verify that a successful run produced exactly the promised files."""
    visible_entries = {
        path.name: path
        for path in output_dir.iterdir()
        if not path.name.startswith(".")
    }
    actual = {
        name for name, path in visible_entries.items() if path.is_file()
    }
    missing = sorted(expected - actual)
    unexpected = sorted(set(visible_entries) - expected)
    if missing or unexpected:
        raise AssertionError(
            "v3 output contract failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(expected):
        if not name.endswith(".png"):
            continue
        path = output_dir / name
        try:
            image = plt.imread(path)
        except Exception as exc:
            raise AssertionError(f"Generated PNG is unreadable: {path}") from exc
        if image.ndim not in (2, 3) or image.size == 0:
            raise AssertionError(f"Generated PNG has invalid image data: {path}")


def evaluate_group(name: str) -> tuple[str, list[str], str]:
    matched = [group for group, rule in SPECIFIC_GROUP_RULES if rule(name)]
    if matched:
        assignment_type = "specific rule" if len(matched) == 1 else "overlap"
        return matched[0], matched, assignment_type
    if name.startswith("MagpieData"):
        return "Other composition", [], "composition fallback"
    return "Other structure", [], "structure fallback"


def build_group_audit(feature_names: list[str]) -> pd.DataFrame:
    name_counts = pd.Series(feature_names, dtype="object").value_counts()
    records = []
    for position, name in enumerate(feature_names):
        assigned, matched, assignment_type = evaluate_group(name)
        records.append(
            {
                "column position": position,
                "feature": name,
                "assigned group": assigned,
                "assignment type": assignment_type,
                "n specific group matches": len(matched),
                "matched specific groups": " | ".join(matched) if matched else "NA",
                "duplicate name count": int(name_counts[name]),
                "is duplicate feature name": bool(name_counts[name] > 1),
                "is BandCenter": is_band_center(name),
            }
        )
    return pd.DataFrame.from_records(records)


def validate_group_audit(
    audit: pd.DataFrame,
    audit_path: Path,
    fail_on_fallback: bool,
) -> None:
    duplicate_mask = audit["is duplicate feature name"]
    overlap_mask = audit["n specific group matches"] > 1
    fallback_mask = audit["assignment type"].str.endswith("fallback")

    problems = []
    if duplicate_mask.any():
        names = audit.loc[duplicate_mask, "feature"].drop_duplicates().tolist()
        problems.append(f"duplicate feature names ({len(names)}): {names[:10]}")
    if overlap_mask.any():
        names = audit.loc[overlap_mask, "feature"].tolist()
        problems.append(f"overlapping specific rules ({len(names)}): {names[:10]}")
    if fail_on_fallback and fallback_mask.any():
        names = audit.loc[fallback_mask, "feature"].tolist()
        problems.append(f"generic fallback assignments ({len(names)}): {names[:10]}")

    must_stop = (
        FAIL_ON_DUPLICATE_OR_OVERLAP
        and (duplicate_mask.any() or overlap_mask.any())
    ) or (fail_on_fallback and fallback_mask.any())
    if must_stop:
        raise ValueError(
            "Feature-group audit failed. Review the already-written audit CSV at "
            f"{audit_path}:\n  - " + "\n  - ".join(problems)
        )


def load_shap_importance(
    shap_dir: Path,
    model: str,
    folds: list[int],
    n_features: int,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    per_fold: dict[str, np.ndarray] = {}
    fold_columns = []
    sample_counts = []

    for fold in folds:
        path = shap_dir / f"shap_values_{model}_fold_{fold}.npy"
        values = np.load(path)
        if values.ndim != 2:
            raise ValueError(f"{path}: expected 2-D SHAP array, got {values.shape}")
        if values.shape[1] != n_features:
            raise ValueError(
                f"{path}: n columns {values.shape[1]} != feature names {n_features}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: SHAP values contain NaN or infinity.")

        column = f"fold_{fold}"
        fold_columns.append(column)
        per_fold[column] = np.abs(values).mean(axis=0)
        sample_counts.append(values.shape[0])
        print(f"  fold {fold}: {values.shape[0]} samples")

    return pd.DataFrame(per_fold), fold_columns, sample_counts


def aggregate_group_shares(
    feature_frame: pd.DataFrame,
    fold_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    唯一正式组占比口径（与精确 reference anchors 一致）：
      1. 每折组内 mean|SHAP| 求和；
      2. 每折除以该折全部特征的 mean|SHAP| 总和；
      3. 对五折占比等权平均。
    """
    group_raw_by_fold = feature_frame.groupby("group", sort=False)[fold_columns].sum()
    fold_total_importance = feature_frame[fold_columns].sum(axis=0)
    if (fold_total_importance <= 0).any():
        bad = fold_total_importance[fold_total_importance <= 0].index.tolist()
        raise ValueError(f"Total mean|SHAP| is not positive in: {bad}")

    group_share_by_fold = group_raw_by_fold.div(fold_total_importance, axis=1) * 100
    final_group_share = group_share_by_fold.mean(axis=1)
    group_counts = feature_frame.groupby("group", sort=False).size()
    group_summary = pd.DataFrame(
        {
            "n columns": group_counts,
            "Group total SHAP share (%)": final_group_share,
            "Fold-to-fold share std (%)": group_share_by_fold.std(axis=1),
        }
    )
    group_summary["Mean share per column (%/column)"] = (
        group_summary["Group total SHAP share (%)"] / group_summary["n columns"]
    )
    group_summary.index.name = "Feature group"
    group_summary = group_summary.sort_values(
        "Group total SHAP share (%)", ascending=False
    )

    if not np.allclose(group_share_by_fold.sum(axis=0).to_numpy(), 100.0):
        raise AssertionError("At least one fold's group shares do not sum to 100%.")
    if not np.isclose(group_summary["Group total SHAP share (%)"].sum(), 100.0):
        raise AssertionError("Mean group shares do not sum to 100%.")

    return group_summary, group_share_by_fold, fold_total_importance


def calculate_structural_total(
    feature_frame: pd.DataFrame,
    fold_columns: list[str],
    fold_total_importance: pd.Series,
) -> dict[str, object]:
    structural_mask = feature_frame["group"].isin(STRUCTURAL_GROUPS)
    band_center_mask = feature_frame["feature"].map(is_band_center)
    if INCLUDE_BAND_CENTER_IN_STRUCTURAL_TOTAL:
        structural_mask = structural_mask | band_center_mask

    raw_by_fold = feature_frame.loc[structural_mask, fold_columns].sum(axis=0)
    share_by_fold = raw_by_fold / fold_total_importance * 100
    final_share = share_by_fold.mean()
    major_mask = feature_frame["group"].isin(MAJOR_STRUCTURAL_GROUPS)
    major_raw_by_fold = feature_frame.loc[major_mask, fold_columns].sum(axis=0)
    major_share_by_fold = major_raw_by_fold / fold_total_importance * 100
    final_major_share = major_share_by_fold.mean()
    secondary_mask = feature_frame["group"].isin(
        ("Dimensionality", "Other structure")
    )
    secondary_raw_by_fold = feature_frame.loc[
        secondary_mask, fold_columns
    ].sum(axis=0)
    secondary_share_by_fold = secondary_raw_by_fold / fold_total_importance * 100
    final_secondary_share = secondary_share_by_fold.mean()
    return {
        "n columns": int(structural_mask.sum()),
        "Group total SHAP share (%)": float(final_share),
        "Fold-to-fold share std (%)": float(share_by_fold.std()),
        "Mean share per column (%/column)": (
            float(final_share) / int(structural_mask.sum())
            if structural_mask.sum() > 0
            else np.nan
        ),
        "BandCenter columns found": int(band_center_mask.sum()),
        "BandCenter included": INCLUDE_BAND_CENTER_IN_STRUCTURAL_TOTAL,
        "Major structural share (%)": float(final_major_share),
        "Dimensionality + Other structure share (%)": float(
            final_secondary_share
        ),
        "structural groups": " | ".join(STRUCTURAL_GROUPS),
        "selected features": " | ".join(
            feature_frame.loc[structural_mask, "feature"].tolist()
        ),
    }


def rounded_match(actual: float, target: float, decimals: int) -> bool:
    if not np.isfinite(actual):
        return False
    return bool(round(float(actual), decimals) == target)


def within_reference_tolerance(actual: float, target: float) -> bool:
    if not np.isfinite(actual):
        return False
    return bool(abs(float(actual) - float(target)) < REFERENCE_ABS_TOLERANCE_PCT)


def build_reference_checks(
    group_summary: pd.DataFrame,
    structural: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks = [
        (
            "BandCenter (L2 only)",
            "n columns",
            REFERENCE_BANDCENTER_COLUMNS,
            int(group_summary.loc["BandCenter (L2 only)", "n columns"]),
            0,
        ),
        (
            "BandCenter (L2 only)",
            "Group SHAP share (%)",
            REFERENCE_BANDCENTER_SHARE,
            float(
                group_summary.loc[
                    "BandCenter (L2 only)", "Group total SHAP share (%)"
                ]
            ),
            REFERENCE_DECIMALS,
        ),
        (
            "Ionicity",
            "n columns",
            REFERENCE_IONICITY_COLUMNS,
            int(group_summary.loc["Ionicity", "n columns"]),
            0,
        ),
        (
            "Ionicity",
            "Group SHAP share (%)",
            REFERENCE_IONICITY_SHARE,
            float(group_summary.loc["Ionicity", "Group total SHAP share (%)"]),
            REFERENCE_DECIMALS,
        ),
        (
            "Ionicity",
            "Mean share/column (%/column)",
            REFERENCE_IONICITY_DENSITY,
            float(
                group_summary.loc[
                    "Ionicity", "Mean share per column (%/column)"
                ]
            ),
            REFERENCE_DECIMALS,
        ),
    ]
    group_records = []
    for group, metric, target, actual, decimals in checks:
        is_integer_check = decimals == 0
        difference = abs(float(actual) - float(target))
        match = (
            float(actual) == float(target)
            if is_integer_check
            else within_reference_tolerance(float(actual), float(target))
        )
        group_records.append(
            {
                "Feature group": group,
                "metric": metric,
                "reference": target,
                "calculated": actual,
                "rounding decimals": decimals,
                "absolute difference": difference,
                "absolute tolerance": 0.0 if is_integer_check else REFERENCE_ABS_TOLERANCE_PCT,
                "status": "MATCH" if match else "NO MATCH",
            }
        )

    structural_share = float(structural["Group total SHAP share (%)"])
    major_share = float(structural["Major structural share (%)"])
    secondary_share = float(
        structural["Dimensionality + Other structure share (%)"]
    )
    total_match = within_reference_tolerance(
        structural_share, REFERENCE_STRUCTURAL_TOTAL
    )
    major_match = rounded_match(
        major_share,
        REFERENCE_MAJOR_STRUCTURAL_ROUNDED,
        1,
    )
    secondary_match = rounded_match(
        secondary_share,
        REFERENCE_SECONDARY_STRUCTURAL_ROUNDED,
        1,
    )
    structural_record = {
        "aggregation method": AGGREGATION_METHOD,
        "calculated structural total (%)": structural_share,
        "reference structural total (%)": REFERENCE_STRUCTURAL_TOTAL,
        "complete total absolute difference": abs(
            structural_share - REFERENCE_STRUCTURAL_TOTAL
        ),
        "complete total absolute tolerance": REFERENCE_ABS_TOLERANCE_PCT,
        "complete total status": "MATCH" if total_match else "NO MATCH",
        "n columns": structural["n columns"],
        "BandCenter included": structural["BandCenter included"],
        "major four structural groups (%)": major_share,
        "major four rounded reference (%)": REFERENCE_MAJOR_STRUCTURAL_ROUNDED,
        "major four status": "MATCH" if major_match else "NO MATCH",
        "Dimensionality + Other structure (%)": secondary_share,
        "secondary rounded reference (%)": REFERENCE_SECONDARY_STRUCTURAL_ROUNDED,
        "secondary status": "MATCH" if secondary_match else "NO MATCH",
        "status": "MATCH" if total_match and major_match and secondary_match else "NO MATCH",
    }
    return pd.DataFrame(group_records), pd.DataFrame([structural_record])


def build_l1_reference_checks(
    group_summary: pd.DataFrame,
    structural: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build meaningful L1-only checks instead of applying L2 anchors to L1."""
    band_count = int(
        group_summary.loc["BandCenter (L2 only)", "n columns"]
        if "BandCenter (L2 only)" in group_summary.index
        else 0
    )
    ionic_count = int(
        group_summary.loc["Ionicity", "n columns"]
        if "Ionicity" in group_summary.index
        else 0
    )
    total_share = float(group_summary["Group total SHAP share (%)"].sum())
    checks = (
        ("All mutually exclusive groups", "Group SHAP shares sum (%)", 100.0, total_share),
        ("BandCenter (L2 only)", "n columns", 0.0, float(band_count)),
        ("Ionicity", "n columns", float(REFERENCE_IONICITY_COLUMNS), float(ionic_count)),
    )
    records = []
    for group, metric, reference, calculated in checks:
        is_count = metric == "n columns"
        tolerance = 0.0 if is_count else REFERENCE_ABS_TOLERANCE_PCT
        difference = abs(calculated - reference)
        match = difference == 0.0 if is_count else difference < tolerance
        records.append(
            {
                "Feature group": group,
                "metric": metric,
                "reference": reference,
                "calculated": calculated,
                "absolute difference": difference,
                "absolute tolerance": tolerance,
                "status": "MATCH" if match else "NO MATCH",
                "scope": "L1 composition-only audit",
            }
        )

    structural_share = float(structural["Group total SHAP share (%)"])
    structural_columns = int(structural["n columns"])
    structural_match = (
        abs(structural_share) < REFERENCE_ABS_TOLERANCE_PCT
        and structural_columns == 0
    )
    structural_record = {
        "aggregation method": AGGREGATION_METHOD,
        "scope": "L1 composition-only audit",
        "calculated structural total (%)": structural_share,
        "reference L1 structural total (%)": 0.0,
        "absolute difference": abs(structural_share),
        "absolute tolerance": REFERENCE_ABS_TOLERANCE_PCT,
        "n columns": structural_columns,
        "reference n columns": 0,
        "BandCenter included": structural["BandCenter included"],
        "status": "MATCH" if structural_match else "NO MATCH",
    }
    return pd.DataFrame(records), pd.DataFrame([structural_record])


MAE_MEAN_PATTERN = re.compile(
    r"['\"]mae['\"]\s*:\s*\{\s*['\"]mean['\"]\s*:\s*"
    r"(?:np\.float64\()?([-+0-9.eE]+)\)?"
)


def read_mean_mae(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = MAE_MEAN_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not parse mean MAE from {path}")
    value = float(match.group(1))
    if not np.isfinite(value):
        raise ValueError(f"Non-finite mean MAE in {path}: {value}")
    return value


def build_deletion_results(
    full_l2_score: Path,
    symmetry_score: Path,
    coordination_score: Path,
) -> pd.DataFrame:
    full_mae = read_mean_mae(full_l2_score)
    rows = []
    for group, path in (
        ("Global symmetry", symmetry_score),
        ("Coordination fingerprint", coordination_score),
    ):
        deleted_mae = read_mean_mae(path)
        delta = deleted_mae - full_mae
        rows.append(
            {
                "Feature group": group,
                "MAE full L2": full_mae,
                "MAE group deleted": deleted_mae,
                "Deletion ΔMAE": delta,
                "definition": "MAE(group deleted) - MAE(full L2)",
                "interpretation": (
                    "deletion worsens MAE" if delta > 0 else
                    "deletion improves MAE" if delta < 0 else
                    "no change"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_paper_summary(
    group_summary: pd.DataFrame,
    structural: dict[str, object],
    deletion_results: pd.DataFrame,
) -> pd.DataFrame:
    summary = group_summary.reset_index()[
        [
            "Feature group",
            "n columns",
            "Group total SHAP share (%)",
            "Mean share per column (%/column)",
        ]
    ].copy()
    summary = summary.rename(
        columns={
            "Group total SHAP share (%)": "Group SHAP share",
            "Mean share per column (%/column)": "Mean share/column",
        }
    )
    deletion_map = dict(
        zip(deletion_results["Feature group"], deletion_results["Deletion ΔMAE"])
    )
    summary["Deletion ΔMAE"] = summary["Feature group"].map(deletion_map)

    structural_row = pd.DataFrame(
        [
            {
                "Feature group": "Structural total",
                "n columns": structural["n columns"],
                "Group SHAP share": structural[
                    "Group total SHAP share (%)"
                ],
                "Mean share/column": structural[
                    "Mean share per column (%/column)"
                ],
                # 不能把各组 deletion ΔMAE 相加，也不能拿 SHAP 占比代替。
                "Deletion ΔMAE": np.nan,
            }
        ]
    )
    return pd.concat([summary, structural_row], ignore_index=True)


def load_level_feature_frame(
    shap_dir: Path,
    model: str,
    folds: list[int],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, list[str], list[int]]:
    names_path = shap_dir / f"shap_feature_names_{model}.json"
    with names_path.open(encoding="utf-8") as handle:
        feature_names = [str(name) for name in json.load(handle)]

    audit = build_group_audit(feature_names)
    fold_frame, fold_columns, sample_counts = load_shap_importance(
        shap_dir, model, folds, len(feature_names)
    )
    feature_frame = pd.concat(
        [
            audit[["feature", "assigned group"]].rename(
                columns={"assigned group": "group"}
            ),
            fold_frame,
        ],
        axis=1,
    )
    feature_frame["mean"] = feature_frame[fold_columns].mean(axis=1)
    feature_frame["std"] = feature_frame[fold_columns].std(axis=1)
    feature_frame["rank"] = (
        feature_frame["mean"].rank(method="first", ascending=False).astype(int)
    )
    feature_frame = feature_frame.sort_values("rank")
    return feature_names, audit, feature_frame, fold_columns, sample_counts


def validate_l2_partition(
    feature_names: list[str],
    audit: pd.DataFrame,
) -> None:
    if len(feature_names) != EXPECTED_L2_COLUMNS:
        raise ValueError(
            f"L2 must contain {EXPECTED_L2_COLUMNS} columns, got "
            f"{len(feature_names)}."
        )
    if len(audit) != EXPECTED_L2_COLUMNS or audit["assigned group"].isna().any():
        raise ValueError("L2 grouping has an omission or row-count mismatch.")
    if audit["column position"].nunique() != EXPECTED_L2_COLUMNS:
        raise ValueError("L2 column positions are not mutually unique.")

    counts = audit.groupby("assigned group").size()
    band_count = int(counts.get("BandCenter (L2 only)", 0))
    ionic_count = int(counts.get("Ionicity", 0))
    if band_count != REFERENCE_BANDCENTER_COLUMNS:
        raise ValueError(
            f"BandCenter (L2 only) must contain exactly 1 column, got {band_count}."
        )
    if ionic_count != REFERENCE_IONICITY_COLUMNS:
        ionic_names = audit.loc[
            audit["assigned group"] == "Ionicity", "feature"
        ].tolist()
        raise ValueError(
            "Ionicity must contain exactly six Electronegativity features; "
            f"got {ionic_count}: {ionic_names}"
        )


def validate_l1_partition(
    feature_names: list[str],
    audit: pd.DataFrame,
) -> None:
    """Ensure the L1 input is composition-only and has the shared six ionicity columns."""
    if len(audit) != len(feature_names) or audit["assigned group"].isna().any():
        raise ValueError("L1 grouping has an omission or row-count mismatch.")
    if audit["column position"].nunique() != len(feature_names):
        raise ValueError("L1 column positions are not mutually unique.")

    band_names = audit.loc[audit["is BandCenter"], "feature"].tolist()
    if band_names:
        raise ValueError(
            "BandCenter is L2-only but was found in the L1 feature list: "
            f"{band_names}"
        )
    non_composition = audit.loc[
        ~audit["assigned group"].isin(COMPOSITION_GROUPS),
        ["feature", "assigned group"],
    ]
    if not non_composition.empty:
        raise ValueError(
            "L1 must contain composition features only; unexpected rows: "
            f"{non_composition.head(10).to_dict(orient='records')}"
        )
    ionic_names = audit.loc[
        audit["assigned group"] == "Ionicity", "feature"
    ].tolist()
    if len(ionic_names) != REFERENCE_IONICITY_COLUMNS:
        raise ValueError(
            "L1 Ionicity must contain exactly six Electronegativity features; "
            f"got {len(ionic_names)}: {ionic_names}"
        )


def build_attribution_transfer(
    l1_feature_frame: pd.DataFrame,
    l2_feature_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    比较 L1/L2 共有 composition features 的绝对 mean(|SHAP|)。

    A_level(group) = sum over common group features of
                     mean_folds(mean_samples(|SHAP|)).
    Transfer (%) = 100 * (A_L2 - A_L1) / A_L1.
    """
    l1 = l1_feature_frame.set_index("feature", drop=False)
    l2 = l2_feature_frame.set_index("feature", drop=False)

    l1_ionic = set(l1.loc[l1["group"] == "Ionicity", "feature"])
    l2_ionic = set(l2.loc[l2["group"] == "Ionicity", "feature"])
    if len(l1_ionic) != REFERENCE_IONICITY_COLUMNS:
        raise ValueError(
            f"L1 Ionicity must contain six features, got {sorted(l1_ionic)}"
        )
    if len(l2_ionic) != REFERENCE_IONICITY_COLUMNS:
        raise ValueError(
            f"L2 Ionicity must contain six features, got {sorted(l2_ionic)}"
        )
    if l1_ionic != l2_ionic:
        raise ValueError(
            "L1 and L2 Ionicity feature sets differ. "
            f"L1-only={sorted(l1_ionic - l2_ionic)}, "
            f"L2-only={sorted(l2_ionic - l1_ionic)}"
        )

    l1_comp = set(l1.loc[l1["group"].isin(COMPOSITION_GROUPS), "feature"])
    l2_comp = set(l2.loc[l2["group"].isin(COMPOSITION_GROUPS), "feature"])
    common = sorted((l1_comp & l2_comp) - {"band center"})
    if not common:
        raise ValueError("No common L1/L2 composition features were found.")
    if "band center" in common:
        raise AssertionError("BandCenter was not excluded from transfer.")

    records = []
    for feature in common:
        l1_group = str(l1.at[feature, "group"])
        l2_group = str(l2.at[feature, "group"])
        if l1_group != l2_group:
            raise ValueError(
                f"Common feature {feature!r} changes group: "
                f"L1={l1_group}, L2={l2_group}"
            )
        a_l1 = float(l1.at[feature, "mean"])
        a_l2 = float(l2.at[feature, "mean"])
        change = np.nan if a_l1 == 0 else 100 * (a_l2 - a_l1) / a_l1
        records.append(
            {
                "feature": feature,
                "Feature group": l1_group,
                "A_L1 = absolute mean(|SHAP|)": a_l1,
                "A_L2 = absolute mean(|SHAP|)": a_l2,
                "L1 denominator": a_l1,
                "transfer change (%)": change,
                "formula": "100 * (A_L2 - A_L1) / A_L1",
            }
        )
    feature_transfer = pd.DataFrame(records)

    group_transfer = (
        feature_transfer.groupby("Feature group", sort=False)
        .agg(
            **{
                "n common columns": ("feature", "size"),
                "A_L1 = sum absolute mean(|SHAP|)": (
                    "A_L1 = absolute mean(|SHAP|)",
                    "sum",
                ),
                "A_L2 = sum absolute mean(|SHAP|)": (
                    "A_L2 = absolute mean(|SHAP|)",
                    "sum",
                ),
            }
        )
        .reset_index()
    )
    group_transfer["L1 denominator"] = group_transfer[
        "A_L1 = sum absolute mean(|SHAP|)"
    ]
    group_transfer["transfer change (%)"] = 100 * (
        group_transfer["A_L2 = sum absolute mean(|SHAP|)"]
        - group_transfer["A_L1 = sum absolute mean(|SHAP|)"]
    ) / group_transfer["A_L1 = sum absolute mean(|SHAP|)"]
    group_transfer["formula"] = "100 * (A_L2 - A_L1) / A_L1"

    ionic_row = group_transfer.loc[group_transfer["Feature group"] == "Ionicity"]
    if len(ionic_row) != 1:
        raise ValueError("Transfer summary must contain exactly one Ionicity row.")
    ionic_change = float(ionic_row.iloc[0]["transfer change (%)"])
    metadata = {
        "n common composition features": len(common),
        "BandCenter excluded": True,
        "Ionicity feature count": len(l1_ionic),
        "Ionicity features": " | ".join(sorted(l1_ionic)),
        "Ionicity transfer change (%)": ionic_change,
        "Original -20.1% holds after rounding": rounded_match(
            ionic_change, REFERENCE_IONICITY_TRANSFER_ROUNDED, 1
        ),
        "formula": "100 * (A_L2 - A_L1) / A_L1",
    }
    return feature_transfer, group_transfer, metadata


def plot_top_features(
    feature_frame: pd.DataFrame,
    output_path: Path,
    title_tag: str,
) -> None:
    top = feature_frame.nsmallest(TOPK_PLOT, "rank").iloc[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(
        y,
        top["mean"],
        xerr=top["std"],
        color=[GROUP_COLORS.get(group, "#999999") for group in top["group"]],
        edgecolor="white",
        error_kw={"lw": 1, "capsize": 2, "color": "#333333"},
    )
    ax.set_yticks(y, labels=top["feature"])
    ax.set_xlabel("mean(|SHAP|) across official folds")
    ax.set_title(f"{title_tag} — corrected cross-fold SHAP importance")

    seen = list(dict.fromkeys(top["group"]))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS.get(group, "#999999"))
        for group in seen
    ]
    ax.legend(handles, seen, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_group_contribution(
    group_summary: pd.DataFrame,
    output_path: Path,
    title_tag: str,
) -> None:
    ordered = group_summary.sort_values(
        "Group total SHAP share (%)", ascending=True
    )
    y = np.arange(len(ordered))
    fig_height = max(5.0, 0.42 * len(ordered) + 1.8)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    shares = ordered["Group total SHAP share (%)"].astype(float)
    errors = (
        ordered["Fold-to-fold share std (%)"]
        .fillna(0.0)
        .astype(float)
        .clip(lower=0.0)
    )
    ax.barh(
        y,
        shares,
        xerr=errors,
        color=[GROUP_COLORS.get(group, "#999999") for group in ordered.index],
        edgecolor="white",
        error_kw={"lw": 1, "capsize": 2, "color": "#333333"},
    )
    ax.set_yticks(y, labels=ordered.index)
    ax.set_xlabel("Group total SHAP share (%)")
    ax.set_title(
        f"{title_tag} — group contribution\n"
        "normalize within each fold, then average the five folds equally"
    )
    # 标签必须从误差棒的右端开始，而不是从条形末端开始；否则大组的
    # 百分比文字会被水平误差棒穿过。额外扩展 x 轴，为最长标签留空间。
    errorbar_right = shares + errors
    max_errorbar_right = float(errorbar_right.max())
    label_padding = max(max_errorbar_right * 0.012, 0.04)
    label_room = max(max_errorbar_right * 0.20, 1.0)
    ax.set_xlim(left=0.0, right=max_errorbar_right + label_room)
    for row_number, (_, row) in enumerate(ordered.iterrows()):
        label_x = (
            float(row["Group total SHAP share (%)"])
            + float(errors.iloc[row_number])
            + label_padding
        )
        ax.text(
            label_x,
            row_number,
            f"{row['Group total SHAP share (%)']:.1f}%  "
            f"(n={int(row['n columns'])})",
            va="center",
            ha="left",
            fontsize=8,
            color="#444444",
            clip_on=False,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def manuscript_sentences(structural: dict[str, object]) -> tuple[str, str]:
    major = float(structural["Major structural share (%)"])
    secondary = float(
        structural["Dimensionality + Other structure share (%)"]
    )
    total = float(structural["Group total SHAP share (%)"])
    english = (
        f"The four major structural groups together account for {major:.1f}% of "
        f"the total SHAP attribution, while Dimensionality and Other structure "
        f"contribute a further {secondary:.1f}%, giving a complete structural "
        f"total of {total:.1f}%."
    )
    chinese = (
        f"四个主要结构组合计 {major:.1f}%，Dimensionality 和 Other structure "
        f"再贡献约 {secondary:.1f}%，因此完整结构总量为 {total:.1f}%。"
    )
    return english, chinese


def write_report(
    path: Path,
    args: argparse.Namespace,
    folds: list[int],
    l1_sample_counts: list[int],
    l2_sample_counts: list[int],
    audit: pd.DataFrame,
    structural: dict[str, object],
    reference_checks: pd.DataFrame,
    structural_check: pd.DataFrame,
    deletion_results: pd.DataFrame,
    transfer_metadata: dict[str, object],
    group_transfer: pd.DataFrame,
) -> None:
    duplicate_count = int(audit["is duplicate feature name"].sum())
    overlap_count = int((audit["n specific group matches"] > 1).sum())
    fallback_counts = (
        audit.loc[audit["assignment type"].str.endswith("fallback"), "assignment type"]
        .value_counts()
        .to_dict()
    )
    band_center_groups = (
        audit.loc[audit["is BandCenter"], "assigned group"].drop_duplicates().tolist()
    )
    english_sentence, chinese_sentence = manuscript_sentences(structural)

    lines = [
        "Corrected L2 SHAP aggregation and L1-to-L2 transfer audit",
        "========================================================",
        f"L1 SHAP directory: {args.l1_shap_dir}",
        f"L2 SHAP directory: {args.l2_shap_dir}",
        f"Model: {args.model}",
        f"Folds: {folds}",
        f"L1 samples by fold: {l1_sample_counts}",
        f"L2 samples by fold: {l2_sample_counts}",
        f"Aggregation method: {AGGREGATION_METHOD}",
        "",
        "Grouping audit",
        "--------------",
        f"L2 feature columns: {len(audit)} / expected {EXPECTED_L2_COLUMNS}",
        f"Duplicate feature-name rows: {duplicate_count}",
        f"Overlapping specific-rule rows: {overlap_count}",
        f"Generic fallback counts: {fallback_counts}",
        f"BandCenter assigned group(s): {band_center_groups or ['not found']}",
        "Structural groups: " + " | ".join(STRUCTURAL_GROUPS),
        "BandCenter included in structural total: "
        + str(structural["BandCenter included"]),
        "",
        "Deletion ablation",
        "-----------------",
        "Definition: Deletion ΔMAE = MAE(group deleted) - MAE(full L2).",
        "Positive means deletion worsens MAE; negative means deletion improves MAE.",
        deletion_results.to_string(index=False),
        "Only symmetry and coordination receive deletion values; all other "
        "groups remain NA in the final table.",
        "",
        "Attribution transfer",
        "--------------------",
        "Only exact-name composition features common to L1 and L2 are compared.",
        "BandCenter is excluded. Ionicity must be the same six "
        "Electronegativity features in both levels.",
        "A_level(group) = sum over common group columns of "
        "mean_folds(mean_samples(|SHAP|)).",
        "Transfer (%) = 100 * (A_L2 - A_L1) / A_L1; A_L1 is the denominator.",
        f"Transfer metadata: {transfer_metadata}",
        group_transfer.to_string(index=False),
        "",
        "Interpretation guardrails",
        "-------------------------",
        "Mean share/column = group share / n columns. It is an attribution "
        "density descriptor, not a corrected or causal attribution.",
        "Structural total is a subtotal and must not be added again when checking "
        "whether mutually exclusive groups sum to 100%.",
        "SHAP share is never substituted for Deletion ΔMAE.",
        "",
        "Reference group checks",
        "----------------------",
        reference_checks.to_string(index=False),
        "",
        "Structural-total check",
        "----------------------",
        structural_check.to_string(index=False),
        "",
        "Suggested manuscript sentence",
        "-----------------------------",
        english_sentence,
        chinese_sentence,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_l1_report(
    path: Path,
    args: argparse.Namespace,
    folds: list[int],
    sample_counts: list[int],
    audit: pd.DataFrame,
    structural: dict[str, object],
    reference_checks: pd.DataFrame,
    structural_check: pd.DataFrame,
) -> None:
    """Write the full standalone L1 audit report retained from the old workflow."""
    duplicate_count = int(audit["is duplicate feature name"].sum())
    overlap_count = int((audit["n specific group matches"] > 1).sum())
    fallback_counts = (
        audit.loc[
            audit["assignment type"].str.endswith("fallback"),
            "assignment type",
        ]
        .value_counts()
        .to_dict()
    )
    ionic_names = audit.loc[
        audit["assigned group"] == "Ionicity", "feature"
    ].tolist()
    lines = [
        "Corrected L1 composition SHAP aggregation audit",
        "===============================================",
        f"L1 SHAP directory: {args.l1_shap_dir}",
        f"Model: {args.model}",
        f"Folds: {folds}",
        f"Samples by fold: {sample_counts}",
        f"Feature columns: {len(audit)}",
        f"Aggregation method: {AGGREGATION_METHOD}",
        "",
        "Grouping audit",
        "--------------",
        f"Duplicate feature-name rows: {duplicate_count}",
        f"Overlapping specific-rule rows: {overlap_count}",
        f"Generic fallback counts: {fallback_counts}",
        f"BandCenter columns: {int(audit['is BandCenter'].sum())} (expected 0)",
        f"Ionicity columns: {len(ionic_names)} (expected 6)",
        "Ionicity features: " + " | ".join(ionic_names),
        "",
        "Interpretation guardrails",
        "-------------------------",
        "Group shares are normalized within each fold and then averaged with "
        "equal fold weight.",
        "Mean share/column is an attribution-density descriptor, not a causal "
        "effect.",
        "Deletion ablation is an L2-only analysis; all L1 Deletion ΔMAE values "
        "are NA.",
        "",
        "L1 reference checks",
        "-------------------",
        reference_checks.to_string(index=False),
        "",
        "L1 structural-total check",
        "-------------------------",
        structural_check.to_string(index=False),
        f"Calculated structural columns: {structural['n columns']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds)
    output_contract = expected_output_names(args)
    guard_output_directory(args.output_dir, output_contract)

    print("Loading frozen L2 SHAP arrays...")
    (
        l2_feature_names,
        l2_group_audit,
        l2_feature_frame,
        l2_fold_columns,
        l2_sample_counts,
    ) = load_level_feature_frame(args.l2_shap_dir, args.model, folds)
    print(f"L2 feature count: {len(l2_feature_names)}")
    print(f"Aggregation method: {AGGREGATION_METHOD}")

    audit_path = args.output_dir / f"{args.v2_prefix}_feature_group_audit.csv"
    l2_group_audit.to_csv(audit_path, index=False, na_rep="NA")
    print(f"→ {audit_path}")
    validate_group_audit(l2_group_audit, audit_path, args.fail_on_fallback)
    validate_l2_partition(l2_feature_names, l2_group_audit)

    full_path = args.output_dir / f"{args.v2_prefix}_shap_crossfold_full.csv"
    l2_feature_frame.to_csv(
        full_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {full_path}")

    group_summary, group_share_by_fold, fold_total_importance = (
        aggregate_group_shares(l2_feature_frame, l2_fold_columns)
    )
    group_path = args.output_dir / f"{args.v2_prefix}_shap_group_contribution.csv"
    group_summary.to_csv(group_path, float_format="%.12f", na_rep="NA")
    print(f"→ {group_path}")

    fold_share_path = args.output_dir / f"{args.v2_prefix}_group_share_by_fold.csv"
    group_share_by_fold.to_csv(
        fold_share_path, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {fold_share_path}")

    structural = calculate_structural_total(
        l2_feature_frame, l2_fold_columns, fold_total_importance
    )
    reference_checks, structural_check = build_reference_checks(
        group_summary, structural
    )

    reference_path = args.output_dir / f"{args.v2_prefix}_reference_share_check.csv"
    reference_checks.to_csv(
        reference_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {reference_path}")

    structural_path = args.output_dir / f"{args.v2_prefix}_structural_total_check.csv"
    structural_check.to_csv(
        structural_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {structural_path}")

    deletion_results = build_deletion_results(
        args.full_l2_score,
        args.symmetry_score,
        args.coordination_score,
    )
    deletion_path = args.output_dir / f"{args.v2_prefix}_deletion_ablation.csv"
    deletion_results.to_csv(
        deletion_path, index=False, float_format="%.15f", na_rep="NA"
    )
    print(f"→ {deletion_path}")

    paper_summary = build_paper_summary(
        group_summary, structural, deletion_results
    )
    paper_path = args.output_dir / f"{args.v2_prefix}_paper_group_summary.csv"
    paper_summary.to_csv(
        paper_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {paper_path}")

    print("\nLoading frozen L1 SHAP arrays for full L1 outputs and transfer...")
    (
        l1_feature_names,
        l1_group_audit,
        l1_feature_frame,
        l1_fold_columns,
        l1_sample_counts,
    ) = load_level_feature_frame(args.l1_shap_dir, args.model, folds)
    print(f"L1 feature count: {len(l1_feature_names)}")
    l1_audit_path = args.output_dir / f"{args.v1_prefix}_feature_group_audit.csv"
    l1_group_audit.to_csv(l1_audit_path, index=False, na_rep="NA")
    print(f"→ {l1_audit_path}")
    validate_group_audit(l1_group_audit, l1_audit_path, args.fail_on_fallback)
    validate_l1_partition(l1_feature_names, l1_group_audit)

    l1_full_path = args.output_dir / f"{args.v1_prefix}_shap_crossfold_full.csv"
    l1_feature_frame.to_csv(
        l1_full_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {l1_full_path}")

    # L1 与 L2 共用同一聚合和防遮挡绘图函数。
    l1_group_summary, l1_group_share_by_fold, l1_fold_total = (
        aggregate_group_shares(l1_feature_frame, l1_fold_columns)
    )
    l1_group_path = args.output_dir / f"{args.v1_prefix}_shap_group_contribution.csv"
    l1_group_summary.to_csv(
        l1_group_path, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {l1_group_path}")
    l1_fold_share_path = (
        args.output_dir / f"{args.v1_prefix}_group_share_by_fold.csv"
    )
    l1_group_share_by_fold.to_csv(
        l1_fold_share_path, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {l1_fold_share_path}")

    l1_structural = calculate_structural_total(
        l1_feature_frame, l1_fold_columns, l1_fold_total
    )
    l1_reference_checks, l1_structural_check = build_l1_reference_checks(
        l1_group_summary, l1_structural
    )
    l1_reference_path = (
        args.output_dir / f"{args.v1_prefix}_reference_share_check.csv"
    )
    l1_reference_checks.to_csv(
        l1_reference_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {l1_reference_path}")
    l1_structural_path = (
        args.output_dir / f"{args.v1_prefix}_structural_total_check.csv"
    )
    l1_structural_check.to_csv(
        l1_structural_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {l1_structural_path}")

    no_l1_deletions = pd.DataFrame(
        columns=["Feature group", "Deletion ΔMAE"]
    )
    l1_paper_summary = build_paper_summary(
        l1_group_summary, l1_structural, no_l1_deletions
    )
    l1_paper_path = (
        args.output_dir / f"{args.v1_prefix}_paper_group_summary.csv"
    )
    l1_paper_summary.to_csv(
        l1_paper_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {l1_paper_path}")

    l1_top_plot_path = (
        args.output_dir / f"{args.v1_prefix}_shap_crossfold_top{TOPK_PLOT}.png"
    )
    plot_top_features(
        l1_feature_frame,
        l1_top_plot_path,
        DEFAULT_L1_TITLE_TAG,
    )
    print(f"→ {l1_top_plot_path}")

    l1_group_plot_path = (
        args.output_dir / f"{args.v1_prefix}_shap_group_contribution.png"
    )
    plot_group_contribution(
        l1_group_summary,
        l1_group_plot_path,
        DEFAULT_L1_TITLE_TAG,
    )
    print(f"→ {l1_group_plot_path}")

    l1_report_path = (
        args.output_dir / f"{args.v1_prefix}_aggregation_audit_report.txt"
    )
    write_l1_report(
        l1_report_path,
        args,
        folds,
        l1_sample_counts,
        l1_group_audit,
        l1_structural,
        l1_reference_checks,
        l1_structural_check,
    )
    print(f"→ {l1_report_path}")

    feature_transfer, group_transfer, transfer_metadata = (
        build_attribution_transfer(l1_feature_frame, l2_feature_frame)
    )
    feature_transfer_path = (
        args.output_dir / f"{args.transfer_prefix}_features.csv"
    )
    feature_transfer.to_csv(
        feature_transfer_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {feature_transfer_path}")
    group_transfer_path = (
        args.output_dir / f"{args.transfer_prefix}_groups.csv"
    )
    group_transfer.to_csv(
        group_transfer_path, index=False, float_format="%.12f", na_rep="NA"
    )
    print(f"→ {group_transfer_path}")
    transfer_metadata_path = (
        args.output_dir / f"{args.transfer_prefix}_metadata.json"
    )
    transfer_metadata_path.write_text(
        json.dumps(transfer_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"→ {transfer_metadata_path}")

    top_plot_path = (
        args.output_dir / f"{args.v2_prefix}_shap_crossfold_top{TOPK_PLOT}.png"
    )
    plot_top_features(l2_feature_frame, top_plot_path, args.title_tag)
    print(f"→ {top_plot_path}")

    group_plot_path = (
        args.output_dir / f"{args.v2_prefix}_shap_group_contribution.png"
    )
    plot_group_contribution(group_summary, group_plot_path, args.title_tag)
    print(f"→ {group_plot_path}")

    english_sentence, chinese_sentence = manuscript_sentences(structural)
    sentence_path = args.output_dir / f"{args.v2_prefix}_manuscript_sentence.txt"
    sentence_path.write_text(
        english_sentence + "\n" + chinese_sentence + "\n",
        encoding="utf-8",
    )
    print(f"→ {sentence_path}")

    report_path = args.output_dir / f"{args.v2_prefix}_aggregation_audit_report.txt"
    write_report(
        report_path,
        args,
        folds,
        l1_sample_counts,
        l2_sample_counts,
        l2_group_audit,
        structural,
        reference_checks,
        structural_check,
        deletion_results,
        transfer_metadata,
        group_transfer,
    )
    print(f"→ {report_path}")

    print("\nReference group checks:")
    print(reference_checks.to_string(index=False))
    print("\nStructural-total check:")
    print(structural_check.to_string(index=False))
    print("\nDeletion ablation:")
    print(deletion_results.to_string(index=False))
    print("\nL1-to-L2 attribution transfer:")
    print(group_transfer.to_string(index=False))
    print(
        "\nBandCenter included in structural total: "
        f"{structural['BandCenter included']}"
    )
    print(
        "Mean share per column is an attribution density descriptor only; "
        "it is not a corrected causal attribution."
    )
    if args.fail_on_reference_mismatch:
        group_ok = (reference_checks["status"] == "MATCH").all()
        structural_ok = (structural_check["status"] == "MATCH").all()
        l1_group_ok = (l1_reference_checks["status"] == "MATCH").all()
        l1_structural_ok = (l1_structural_check["status"] == "MATCH").all()
        if not (
            group_ok and structural_ok and l1_group_ok and l1_structural_ok
        ):
            raise ValueError(
                "Reference-share validation failed. Review the L1/L2 reference "
                "and structural check CSV files."
            )

    verify_output_contract(args.output_dir, output_contract)
    print(
        "\n✅ Complete corrected L1/L2 outputs and transfer completed: "
        f"{len(output_contract)} files in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
