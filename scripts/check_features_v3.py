"""
Release change: hard-coded local paths were replaced by command-line
arguments; all analytical operations are unchanged.
特征缺失/失败率完整检查
=======================
回答三个问题（对应论文 SI 需要的数字）：
  Q1. 每个 featurizer 的样本级失败率（含 NaN 行 / 全 0 行 / 部分缺失行）
  Q2. 跨所有结构 featurizer 的并集：多少样本"至少有一处缺失"
  Q3. 缺失样本的带隙分布是否与整体一致（缺失是否随机）

用法：python scripts/check_features_v3.py \
  --cache-dir /实际位置/matbench_cache
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit structural-featurizer failures across MatBench folds."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing fold_*_{train,test}_*.pkl cache files.",
    )
    return parser.parse_args()

# Q3 标签直接从 matbench task 加载（无需额外文件）。
# 注意：缓存 pkl 的 index 是 0..n-1 位置序号，与 matbench df 行顺序一致，
# 所以标签用 reset_index(drop=True) 做位置对齐。
USE_MATBENCH_LABELS = True
TASK_NAME = "matbench_mp_gap"

STRUCT_FEATURIZERS = [
    "DensityFeatures", "GlobalSymmetryFeatures", "StructuralHeterogeneity",
    "ChemicalOrdering", "Dimensionality", "SiteStatsFingerprint",
]

FNAME_RE = re.compile(r"fold_(\d+)_(train|test)_(.+)\.pkl")


def row_level_stats(df: pd.DataFrame) -> dict:
    """样本级统计：NaN 行、全 0 行、任意缺失行。"""
    num = df.select_dtypes(include=[np.number])
    nan_any = df.isna().any(axis=1)                # 至少一列 NaN
    nan_all = df.isna().all(axis=1)                # 整行 NaN
    zero_all = (num == 0).all(axis=1) & ~nan_any   # 整行全 0（且不是 NaN 行）
    # "问题行" = 有 NaN 或整行全 0（全 0 视为失败填充的痕迹）
    bad = nan_any | zero_all
    return {
        "n": len(df),
        "nan_any_rows": int(nan_any.sum()),
        "nan_all_rows": int(nan_all.sum()),
        "zero_all_rows": int(zero_all.sum()),
        "bad_rows": int(bad.sum()),
        "bad_mask": bad,
    }


def worst_columns(df: pd.DataFrame, topk: int = 5) -> str:
    """NaN 最多的前几列，便于定位是哪个具体特征在失败。"""
    col_nan = df.isna().mean().sort_values(ascending=False)
    col_nan = col_nan[col_nan > 0].head(topk)
    if col_nan.empty:
        return "-"
    return "; ".join(f"{c}:{r:.2%}" for c, r in col_nan.items())


def main(cache_dir: Path):
    cache_dir = cache_dir.expanduser().resolve()
    files = sorted(cache_dir.glob("*.pkl"))
    print(f"共找到 {len(files)} 个缓存文件\n")

    # ---------- Q1: 每个文件的样本级失败率 ----------
    print("=" * 110)
    print("Q1. 各 featurizer 样本级失败率")
    print(f"{'文件名':<52} {'行数':>7} {'NaN行':>7} {'全0行':>7} {'问题行':>7} {'占比':>8}  NaN最多的列")
    print("-" * 110)

    # fold -> split -> {featurizer_name: bad_mask}
    masks = defaultdict(lambda: defaultdict(dict))

    for p in files:
        m = FNAME_RE.match(p.name)
        try:
            df = pd.read_pickle(p)
        except Exception as e:
            print(f"{p.name:<52} 读取失败: {e}")
            continue
        if not isinstance(df, pd.DataFrame):
            print(f"{p.name:<52} 非 DataFrame，跳过")
            continue

        s = row_level_stats(df)
        ratio = s["bad_rows"] / s["n"] if s["n"] else 0.0
        flag = " ⚠️" if ratio > 0.02 else ""
        print(f"{p.name:<52} {s['n']:>7} {s['nan_any_rows']:>7} "
              f"{s['zero_all_rows']:>7} {s['bad_rows']:>7} {ratio:>7.2%}{flag}  "
              f"{worst_columns(df)}")

        if m:
            fold, split, feat = m.group(1), m.group(2), m.group(3)
            if feat in STRUCT_FEATURIZERS:
                masks[fold][split][feat] = s["bad_mask"]

    # ---------- Q2: 跨 featurizer 并集 ----------
    print("\n" + "=" * 110)
    print("Q2. 跨结构 featurizer 并集：至少在一处缺失/失败的样本占比")
    print(f"{'fold/split':<20} {'样本数':>8} {'并集问题样本':>12} {'占比':>8}   各featurizer单独占比")
    print("-" * 110)

    union_bad_index = {}  # (fold, split) -> Index of bad samples，供 Q3 用

    for fold in sorted(masks):
        for split in sorted(masks[fold]):
            feat_masks = masks[fold][split]
            if not feat_masks:
                continue
            # 按 index 对齐取并集
            aligned = pd.DataFrame(feat_masks)  # index 自动对齐，缺的算 NaN
            aligned = aligned.fillna(True)      # 某文件缺该样本 -> 视为问题
            union = aligned.any(axis=1)
            n, nbad = len(union), int(union.sum())
            per_feat = ", ".join(
                f"{f}:{feat_masks[f].mean():.2%}" for f in feat_masks
            )
            print(f"fold{fold}_{split:<13} {n:>8} {nbad:>12} {nbad/n:>7.2%}   {per_feat}")
            union_bad_index[(fold, split)] = union[union].index

    # ---------- Q3: 缺失是否随机（与标签的关系） ----------
    print("\n" + "=" * 110)
    if not USE_MATBENCH_LABELS:
        print("Q3. USE_MATBENCH_LABELS=False，跳过。")
        return
    if not union_bad_index:
        print("Q3. Q2 没有产生任何问题样本索引，无需检查缺失-标签相关性。收工。")
        return

    print("Q3. 缺失样本的带隙分布 vs 整体（标签来自 matbench task）")
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False)
    task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
    task.load()
    target_col = task.metadata["target"]

    def get_labels(fold: str, split: str) -> pd.Series:
        if split == "train":
            df = task.get_train_and_val_data(int(fold), as_type="df")
        else:
            df = task.get_test_data(int(fold), as_type="df", include_target=True)
        # 位置对齐：缓存文件 index 是 0..n-1，顺序与该 df 一致
        return df[target_col].reset_index(drop=True)

    print(f"{'fold/split':<20} {'n_bad':>6} {'bad均值':>8} {'bad中位数':>9} "
          f"{'全体均值':>8} {'全体中位数':>9} {'bad金属占比':>11} {'全体金属占比':>12}")
    print("-" * 110)
    for (fold, split), bad_idx in union_bad_index.items():
        try:
            y = get_labels(fold, split)
        except Exception as e:
            print(f"fold{fold}_{split}: 标签加载失败 ({e})，跳过")
            continue
        idx = bad_idx.intersection(y.index)
        if len(idx) == 0:
            continue
        yb, ya = y.loc[idx], y
        print(f"fold{fold}_{split:<13} {len(idx):>6} {yb.mean():>8.3f} {yb.median():>9.3f} "
              f"{ya.mean():>8.3f} {ya.median():>9.3f} "
              f"{(yb == 0).mean():>10.2%} {(ya == 0).mean():>11.2%}")

    print("\n判读：bad 样本的均值/中位数/金属占比若与全体接近 → 缺失近似随机，"
          "论文里一句话带过即可；若明显偏离 → 需要在 SHAP 解释时排查相关特征。")


if __name__ == "__main__":
    args = parse_args()
    main(args.cache_dir)
