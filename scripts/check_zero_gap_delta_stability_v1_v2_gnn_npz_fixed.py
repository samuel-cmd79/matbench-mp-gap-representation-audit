#!/usr/bin/env python3
"""
check_zero_gap_delta_stability_v1_v2_gnn_npz.py

用途
----
直接从 Matbench 官方 matbench_mp_gap task 获取每个 fold 的真实标签，
检查 zero-gap subset 上三种模型的逐 fold 稳定性：

    v1  : composition-only XGBoost raw predictions (.npy)
    v2  : structure-aware XGBoost raw predictions (.npy)
    GNN : graph model raw predictions (.npz)

GNN NPZ 约定
------------
每个 fold 的 .npz 文件包含：
    preds : 预测值
    ids   : Matbench material IDs

脚本会：
    1. 检查 GNN ids 是否与官方 test fold 完全一致；
    2. 如果集合一致但顺序不同，自动按官方顺序重排；
    3. 如果 ID 集合不同，直接报错停止。

核心指标
--------
对 y_true == 0 的样本：

    delta_v1_v2_i  = abs(v1_raw_i) - abs(v2_raw_i)
    delta_v2_gnn_i = abs(v2_raw_i) - abs(gnn_raw_i)
    delta_v1_gnn_i = abs(v1_raw_i) - abs(gnn_raw_i)

delta > 0 表示后一个模型更接近物理边界 0。

输出
----
1. zero_gap_three_models_by_fold.csv
2. zero_gap_pairwise_stability_by_fold.csv
3. zero_gap_three_models_samples.csv
4. zero_gap_pairwise_pooled.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 用户配置区：只改这里
# ============================================================

# v1 旧 raw XGBoost 预测
V1_RAW_PATTERN = (
    "../matbench_outputs/v1_predictions_{model}/pred_fold_{fold}.npy"
)

# v2 旧 raw XGBoost 预测
V2_RAW_PATTERN = (
    "../matbench_outputs/v2_predictions_{model}/pred_fold_{fold}.npy"
)

# v1/v2 预测所使用的模型名（用于替换路径中的 {model}）
MODEL = "xgb"

# GNN raw prediction，按你现有 pipeline 的格式
GNN_RAW_PATTERN = (
    "../results_v4/fold_{fold}/test_preds.npz"
)

# GNN .npz 内部 key
GNN_PRED_KEY = "preds"
GNN_ID_KEY = "ids"

# Matbench 设置
TASK_NAME = "matbench_mp_gap"
FOLDS = [0, 1, 2, 3, 4]

# 输出文件
OUTPUT_MODEL_FOLD_CSV = Path(
    "zero_gap_three_models_by_fold.csv"
)
OUTPUT_PAIRWISE_FOLD_CSV = Path(
    "zero_gap_pairwise_stability_by_fold.csv"
)
OUTPUT_SAMPLE_CSV = Path(
    "zero_gap_three_models_samples.csv"
)
OUTPUT_POOLED_CSV = Path(
    "zero_gap_pairwise_pooled.csv"
)

# 容差
ZERO_ATOL = 1e-12
TIE_ATOL = 1e-12


# ============================================================
# 以下代码通常不需要修改
# ============================================================


def safe_pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else float("nan")


def load_npy_prediction(
    pattern: str,
    fold: int,
    expected_length: int,
) -> np.ndarray:
    """加载 v1/v2 的 .npy 预测。"""
    path = Path(pattern.format(fold=fold, model=MODEL))

    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    pred = np.load(
        path,
        allow_pickle=False,
    )
    pred = np.asarray(pred).reshape(-1).astype(
        np.float64,
        copy=False,
    )

    if len(pred) != expected_length:
        raise ValueError(
            f"{path.name}: 长度 {len(pred)} "
            f"!= 官方 test 数 {expected_length}"
        )

    if not np.all(np.isfinite(pred)):
        n_bad = int(np.sum(~np.isfinite(pred)))
        raise ValueError(
            f"{path} 中有 {n_bad} 个 NaN 或 inf"
        )

    return pred


def load_gnn_npz_prediction(
    pattern: str,
    fold: int,
    mbid_ref: np.ndarray,
) -> np.ndarray:
    """
    加载 GNN .npz，并用 ids 对官方 Matbench test IDs 校验/重排。
    """
    path = Path(pattern.format(fold=fold))

    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    data = np.load(
        path,
        allow_pickle=True,
    )

    if (
        GNN_PRED_KEY not in data.files
        or GNN_ID_KEY not in data.files
    ):
        raise KeyError(
            f"{path}: 需要 keys "
            f"'{GNN_PRED_KEY}'/'{GNN_ID_KEY}'，"
            f"实际 keys={data.files}"
        )

    pred = np.asarray(
        data[GNN_PRED_KEY]
    ).reshape(-1).astype(
        np.float64,
        copy=False,
    )

    ids = np.asarray(
        [
            str(x)
            for x in np.asarray(
                data[GNN_ID_KEY]
            ).reshape(-1)
        ],
        dtype=object,
    )

    if len(pred) != len(ids):
        raise ValueError(
            f"{path.name}: preds 长度 {len(pred)} "
            f"!= ids 长度 {len(ids)}"
        )

    if len(pred) != len(mbid_ref):
        raise ValueError(
            f"{path.name}: 长度 {len(pred)} "
            f"!= 官方 test 数 {len(mbid_ref)}"
        )

    if not np.all(np.isfinite(pred)):
        n_bad = int(np.sum(~np.isfinite(pred)))
        raise ValueError(
            f"{path} 中有 {n_bad} 个 NaN 或 inf"
        )

    # 顺序一致，直接返回
    if np.array_equal(ids, mbid_ref):
        return pred

    # 集合一致但顺序不同，自动重排
    if set(ids) == set(mbid_ref):
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"{path}: GNN ids 中存在重复项，"
                "无法安全重排"
            )

        index_by_id = pd.Series(
            np.arange(len(ids)),
            index=ids,
        )
        order = index_by_id.loc[
            mbid_ref
        ].to_numpy()

        pred = pred[order]

        print(
            f"  ⚠️ {path.name} fold {fold}: "
            "GNN ID 顺序与官方顺序不同，"
            "已自动按官方顺序重排"
        )
        return pred

    missing_in_gnn = sorted(
        set(mbid_ref) - set(ids)
    )
    extra_in_gnn = sorted(
        set(ids) - set(mbid_ref)
    )

    raise ValueError(
        f"{path}: GNN ids 与官方 test IDs 集合不一致。\n"
        f"  GNN 缺少: {missing_in_gnn[:5]}"
        f"{' ...' if len(missing_in_gnn) > 5 else ''}\n"
        f"  GNN 多出: {extra_in_gnn[:5]}"
        f"{' ...' if len(extra_in_gnn) > 5 else ''}"
    )


def exact_sign_test(
    n_later_better: int,
    n_non_ties: int,
) -> float | None:
    """双侧 exact sign test。没有 scipy 时返回 None。"""
    if n_non_ties <= 0:
        return None

    try:
        from scipy.stats import binomtest
    except ImportError:
        return None

    return float(
        binomtest(
            k=n_later_better,
            n=n_non_ties,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def load_matbench_task():
    """加载 Matbench 官方 task。"""
    try:
        from matbench.bench import MatbenchBenchmark
    except ImportError as exc:
        raise ImportError(
            "未找到 matbench。请在安装了 matbench "
            "的环境中运行此脚本。"
        ) from exc

    mb = MatbenchBenchmark(
        autoload=False
    )

    try:
        task = next(
            task
            for task in mb.tasks
            if task.dataset_name == TASK_NAME
        )
    except StopIteration as exc:
        raise ValueError(
            f"Matbench 中找不到 task：{TASK_NAME}"
        ) from exc

    task.load()
    return task


def distribution_summary(
    values: np.ndarray,
) -> dict[str, float]:
    q05, q25, q50, q75, q95 = np.quantile(
        values,
        [0.05, 0.25, 0.50, 0.75, 0.95],
    )

    return {
        "raw_mae": float(
            np.mean(np.abs(values))
        ),
        "mean_raw_pred": float(
            np.mean(values)
        ),
        "median_raw_pred": float(q50),
        "q05_raw_pred": float(q05),
        "q25_raw_pred": float(q25),
        "q75_raw_pred": float(q75),
        "q95_raw_pred": float(q95),
        "iqr_raw_pred": float(q75 - q25),
        "negative_n": int(
            np.sum(values < 0)
        ),
        "negative_pct": float(
            100.0 * np.mean(values < 0)
        ),
    }


def compare_pair(
    fold: int | str,
    comparison: str,
    earlier: np.ndarray,
    later: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """
    earlier -> later。
    delta > 0 表示 later 更接近 0。
    """
    delta = (
        np.abs(earlier)
        - np.abs(later)
    )

    tie_mask = np.isclose(
        delta,
        0.0,
        rtol=0.0,
        atol=TIE_ATOL,
    )
    later_better_mask = (
        (delta > 0.0)
        & ~tie_mask
    )
    earlier_better_mask = (
        (delta < 0.0)
        & ~tie_mask
    )

    n = len(delta)
    n_later_better = int(
        np.sum(later_better_mask)
    )
    n_earlier_better = int(
        np.sum(earlier_better_mask)
    )
    n_ties = int(np.sum(tie_mask))

    p_value = exact_sign_test(
        n_later_better=n_later_better,
        n_non_ties=(
            n_later_better
            + n_earlier_better
        ),
    )

    row = {
        "fold": fold,
        "comparison": comparison,
        "n_zero_gap": n,
        "later_better_n": n_later_better,
        "later_better_pct": safe_pct(
            n_later_better,
            n,
        ),
        "earlier_better_n": n_earlier_better,
        "earlier_better_pct": safe_pct(
            n_earlier_better,
            n,
        ),
        "tie_n": n_ties,
        "tie_pct": safe_pct(
            n_ties,
            n,
        ),
        "mean_delta": float(
            np.mean(delta)
        ),
        "median_delta": float(
            np.median(delta)
        ),
        "q05_delta": float(
            np.quantile(delta, 0.05)
        ),
        "q95_delta": float(
            np.quantile(delta, 0.95)
        ),
        "sign_test_p": p_value,
    }

    return row, delta


def main() -> None:
    print("=" * 78)
    print(
        "Matbench zero-gap 三模型逐 fold 稳定性检查 "
        "(GNN NPZ + ID 校验)"
    )
    print("=" * 78)

    task = load_matbench_task()
    target_col = task.metadata[
        "target"
    ]

    model_fold_rows: list[dict] = []
    pairwise_fold_rows: list[dict] = []
    sample_rows: list[dict] = []

    pooled = {
        "v1": [],
        "v2": [],
        "gnn": [],
    }

    for fold in FOLDS:
        test_df = task.get_test_data(
            fold,
            as_type="df",
            include_target=True,
        )

        y_true = test_df[
            target_col
        ].to_numpy(
            dtype=float
        )

        mbids = np.asarray(
            [
                str(index)
                for index in test_df.index
            ],
            dtype=object,
        )

        pred_v1 = load_npy_prediction(
            V1_RAW_PATTERN,
            fold,
            len(y_true),
        )
        pred_v2 = load_npy_prediction(
            V2_RAW_PATTERN,
            fold,
            len(y_true),
        )
        pred_gnn = load_gnn_npz_prediction(
            GNN_RAW_PATTERN,
            fold,
            mbids,
        )

        zero_mask = np.isclose(
            y_true,
            0.0,
            rtol=0.0,
            atol=ZERO_ATOL,
        )

        n_zero = int(
            np.sum(zero_mask)
        )

        if n_zero == 0:
            raise ValueError(
                f"Fold {fold}: 没有 zero-gap 样本"
            )

        mbid_zero = mbids[zero_mask]
        v1_zero = pred_v1[zero_mask]
        v2_zero = pred_v2[zero_mask]
        gnn_zero = pred_gnn[zero_mask]

        pooled["v1"].append(v1_zero)
        pooled["v2"].append(v2_zero)
        pooled["gnn"].append(gnn_zero)

        # 每个模型的 fold 统计
        for model_name, values in [
            ("v1", v1_zero),
            ("v2", v2_zero),
            ("gnn", gnn_zero),
        ]:
            stats = distribution_summary(
                values
            )

            row = {
                "fold": fold,
                "model": model_name,
                "n_zero_gap": n_zero,
                **stats,
            }

            # 只有 GNN raw/clip 是同一批预测，可精确算
            if model_name == "gnn":
                clipped = np.maximum(
                    values,
                    0.0,
                )
                clipped_mae = float(
                    np.mean(
                        np.abs(clipped)
                    )
                )
                row["clipped_mae"] = (
                    clipped_mae
                )
                row["clip_mae_gain"] = (
                    stats["raw_mae"]
                    - clipped_mae
                )
            else:
                row["clipped_mae"] = np.nan
                row["clip_mae_gain"] = np.nan

            model_fold_rows.append(row)

        # 逐 fold pairwise
        for comparison, earlier, later in [
            (
                "v1_to_v2",
                v1_zero,
                v2_zero,
            ),
            (
                "v2_to_gnn",
                v2_zero,
                gnn_zero,
            ),
            (
                "v1_to_gnn",
                v1_zero,
                gnn_zero,
            ),
        ]:
            row, _ = compare_pair(
                fold=fold,
                comparison=comparison,
                earlier=earlier,
                later=later,
            )
            pairwise_fold_rows.append(row)

        # 逐样本结果
        delta_v1_v2 = (
            np.abs(v1_zero)
            - np.abs(v2_zero)
        )
        delta_v2_gnn = (
            np.abs(v2_zero)
            - np.abs(gnn_zero)
        )
        delta_v1_gnn = (
            np.abs(v1_zero)
            - np.abs(gnn_zero)
        )

        for (
            mbid,
            p1,
            p2,
            pg,
            d12,
            d2g,
            d1g,
        ) in zip(
            mbid_zero,
            v1_zero,
            v2_zero,
            gnn_zero,
            delta_v1_v2,
            delta_v2_gnn,
            delta_v1_gnn,
        ):
            sample_rows.append({
                "fold": fold,
                "mbid": mbid,
                "y_true": 0.0,
                "v1_raw_pred": float(p1),
                "v2_raw_pred": float(p2),
                "gnn_raw_pred": float(pg),
                "gnn_clipped_pred": float(
                    max(pg, 0.0)
                ),
                "abs_v1": float(abs(p1)),
                "abs_v2": float(abs(p2)),
                "abs_gnn": float(abs(pg)),
                "delta_v1_v2": float(d12),
                "delta_v2_gnn": float(d2g),
                "delta_v1_gnn": float(d1g),
            })

        fold_pair_df = pd.DataFrame(
            pairwise_fold_rows
        )
        current_fold = fold_pair_df[
            fold_pair_df["fold"] == fold
        ]

        v1_v2_pct = float(
            current_fold.loc[
                current_fold[
                    "comparison"
                ] == "v1_to_v2",
                "later_better_pct",
            ].iloc[0]
        )
        v2_gnn_pct = float(
            current_fold.loc[
                current_fold[
                    "comparison"
                ] == "v2_to_gnn",
                "later_better_pct",
            ].iloc[0]
        )

        print(
            f"Fold {fold}: "
            f"n_zero={n_zero:,} | "
            f"v2 better than v1="
            f"{v1_v2_pct:.2f}% | "
            f"GNN better than v2="
            f"{v2_gnn_pct:.2f}%"
        )

    model_fold_df = pd.DataFrame(
        model_fold_rows
    )
    pairwise_fold_df = pd.DataFrame(
        pairwise_fold_rows
    )
    sample_df = pd.DataFrame(
        sample_rows
    )

    pooled_v1 = np.concatenate(
        pooled["v1"]
    )
    pooled_v2 = np.concatenate(
        pooled["v2"]
    )
    pooled_gnn = np.concatenate(
        pooled["gnn"]
    )

    pooled_rows = []

    for comparison, earlier, later in [
        (
            "v1_to_v2",
            pooled_v1,
            pooled_v2,
        ),
        (
            "v2_to_gnn",
            pooled_v2,
            pooled_gnn,
        ),
        (
            "v1_to_gnn",
            pooled_v1,
            pooled_gnn,
        ),
    ]:
        row, _ = compare_pair(
            fold="pooled",
            comparison=comparison,
            earlier=earlier,
            later=later,
        )
        pooled_rows.append(row)

    pooled_df = pd.DataFrame(
        pooled_rows
    )

    print("\n逐 fold 配对稳定性")
    print(
        pairwise_fold_df[
            [
                "fold",
                "comparison",
                "n_zero_gap",
                "later_better_pct",
                "earlier_better_pct",
                "tie_pct",
                "median_delta",
                "sign_test_p",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n五折 pooled 配对结果")
    print(
        pooled_df[
            [
                "comparison",
                "n_zero_gap",
                "later_better_pct",
                "earlier_better_pct",
                "tie_pct",
                "mean_delta",
                "median_delta",
                "sign_test_p",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # pooled 三模型分布
    pooled_model_rows = []

    for model_name, values in [
        ("v1", pooled_v1),
        ("v2", pooled_v2),
        ("gnn", pooled_gnn),
    ]:
        stats = distribution_summary(
            values
        )

        row = {
            "model": model_name,
            "n_zero_gap": len(values),
            **stats,
        }

        if model_name == "gnn":
            clipped = np.maximum(
                values,
                0.0,
            )
            row["clipped_mae"] = float(
                np.mean(
                    np.abs(clipped)
                )
            )
            row["clip_mae_gain"] = (
                stats["raw_mae"]
                - row["clipped_mae"]
            )
        else:
            row["clipped_mae"] = np.nan
            row["clip_mae_gain"] = np.nan

        pooled_model_rows.append(row)

    pooled_model_df = pd.DataFrame(
        pooled_model_rows
    )

    print("\n五折 pooled zero-gap raw 分布")
    print(
        pooled_model_df[
            [
                "model",
                "n_zero_gap",
                "raw_mae",
                "mean_raw_pred",
                "median_raw_pred",
                "iqr_raw_pred",
                "q05_raw_pred",
                "q95_raw_pred",
                "negative_pct",
                "clipped_mae",
                "clip_mae_gain",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # 保存结果
    for output in [
        OUTPUT_MODEL_FOLD_CSV,
        OUTPUT_PAIRWISE_FOLD_CSV,
        OUTPUT_SAMPLE_CSV,
        OUTPUT_POOLED_CSV,
    ]:
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    model_fold_df.to_csv(
        OUTPUT_MODEL_FOLD_CSV,
        index=False,
    )
    pairwise_fold_df.to_csv(
        OUTPUT_PAIRWISE_FOLD_CSV,
        index=False,
    )
    sample_df.to_csv(
        OUTPUT_SAMPLE_CSV,
        index=False,
    )
    pooled_df.to_csv(
        OUTPUT_POOLED_CSV,
        index=False,
    )

    print("\n已保存：")
    print(
        f"  {OUTPUT_MODEL_FOLD_CSV.resolve()}"
    )
    print(
        f"  {OUTPUT_PAIRWISE_FOLD_CSV.resolve()}"
    )
    print(
        f"  {OUTPUT_SAMPLE_CSV.resolve()}"
    )
    print(
        f"  {OUTPUT_POOLED_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
