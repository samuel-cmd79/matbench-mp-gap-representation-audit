"""
Release change: hard-coded local paths were replaced by command-line
arguments; all analytical operations are unchanged.
子集稳健性检查
==============
问题:约10%样本的结构特征因featurizer失败被均值填充,且缺失与标签相关
     (失败样本偏向高带隙非金属)。
检查:v1→v2 的改善在"特征完整"和"特征缺失"两组样本上是否一致。

不需要重训。需要:
  1. 缓存目录(用于重建每个 fold test 集的缺失 mask)
  2. v1 和 v2 每个 fold 的官方 test 预测 npy 文件

用法示例：

python subset_robustness_check.py \
  --cache-dir ../matbench_cache \
  --v1-pattern '../matbench_outputs/v1_predictions_{model}/pred_fold_{fold}.npy' \
  --v2-pattern '../matbench_outputs/v2_predictions_{model}/pred_fold_{fold}.npy'
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# ======== 配置 ========
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Level-1 and Level-2 errors on complete and incomplete feature subsets."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--v1-pattern",
        required=True,
        help="Level-1 prediction pattern containing {model} and {fold}.",
    )
    parser.add_argument(
        "--v2-pattern",
        required=True,
        help="Level-2 prediction pattern containing {model} and {fold}.",
    )
    return parser.parse_args()
MODELS = ["rf", "xgb"]
FOLDS = [0, 1, 2, 3, 4]
TASK_NAME = "matbench_mp_gap"

STRUCT_FEATURIZERS = [
    "DensityFeatures", "GlobalSymmetryFeatures", "StructuralHeterogeneity",
    "ChemicalOrdering", "Dimensionality", "SiteStatsFingerprint",
]
# ======================


def bad_mask_for(fold: int, cache_dir: Path) -> pd.Series:
    """重建 fold test 集的缺失 mask(与 check_features_v3 的 Q2 逻辑一致)。"""
    masks = {}
    for feat in STRUCT_FEATURIZERS:
        p = cache_dir / f"fold_{fold}_test_{feat}.pkl"
        if not p.exists():
            print(f"  ⚠️ 缺缓存文件: {p.name},跳过该 featurizer")
            continue
        df = pd.read_pickle(p)
        num = df.select_dtypes(include=[np.number])
        nan_any = df.isna().any(axis=1)
        zero_all = (num == 0).all(axis=1) & ~nan_any
        masks[feat] = nan_any | zero_all
    aligned = pd.DataFrame(masks).fillna(True)
    return aligned.any(axis=1)


def main():
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False)
    task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
    task.load()
    target_col = task.metadata["target"]

    # 汇总容器: model -> group -> version -> list of abs errors
    from collections import defaultdict
    errs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n_complete_total, n_bad_total = 0, 0

    for fold in FOLDS:
        y = task.get_test_data(fold, as_type="df", include_target=True)[target_col]
        y = y.reset_index(drop=True).to_numpy()
        bad = bad_mask_for(fold, cache_dir).to_numpy()
        if len(bad) != len(y):
            raise ValueError(f"fold {fold}: mask 长度 {len(bad)} != 标签长度 {len(y)}")
        n_complete_total += int((~bad).sum())
        n_bad_total += int(bad.sum())

        for model in MODELS:
            for ver, pattern in [
                ("v1", args.v1_pattern),
                ("v2", args.v2_pattern),
            ]:
                p = Path(pattern.format(fold=fold, model=model))
                if not p.exists():
                    print(f"  ⚠️ 找不到预测文件: {p}")
                    continue
                pred = np.load(p)
                if len(pred) != len(y):
                    raise ValueError(f"{p.name}: 预测长度 {len(pred)} != 标签 {len(y)}")
                ae = np.abs(pred - y)
                errs[model]["complete"][ver].append(ae[~bad])
                errs[model]["incomplete"][ver].append(ae[bad])

    print(f"\n特征完整样本: {n_complete_total}  |  特征缺失样本: {n_bad_total} "
          f"({n_bad_total/(n_bad_total+n_complete_total):.1%})")
    print("\n" + "=" * 90)
    print(f"{'模型':<6} {'子集':<12} {'n':>7} {'v1 MAE':>9} {'v2 MAE':>9} "
          f"{'ΔMAE':>9} {'改善%':>8}")
    print("-" * 90)
    for model in MODELS:
        for group in ["complete", "incomplete"]:
            d = errs[model][group]
            if "v1" not in d or "v2" not in d:
                continue
            e1 = np.concatenate(d["v1"])
            e2 = np.concatenate(d["v2"])
            m1, m2 = e1.mean(), e2.mean()
            print(f"{model:<6} {group:<12} {len(e1):>7} {m1:>9.4f} {m2:>9.4f} "
                  f"{m2-m1:>+9.4f} {(m1-m2)/m1:>+7.1%}")
        print("-" * 90)

    print("""
判读:
  - 两组改善幅度相近          → 增益不依赖缺失模式,结论干净,SI一句话。
  - 完整组改善明显 > 缺失组   → 结构信息只在可计算时起效,合理且可写:
        "improvement is concentrated in samples with complete structural
         features, confirming the gain stems from genuine structural
         information rather than imputation artifacts."
  - 缺失组改善反而更大        → 需要警惕模型在利用填充模式,SHAP解释时
        必须在完整子集上复核 ChemicalOrdering / StructuralHeterogeneity
        相关特征的排名。
""")


if __name__ == "__main__":
    main()
