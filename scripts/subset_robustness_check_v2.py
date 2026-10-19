"""
子集稳健性检查 (v2)
====================
比较 Level-1 与 Level-2 在两个子集上的误差, 两种失败口径各算一遍:
  - 主表 (structural)  : 6 个结构特征器中任一个 NaN 或整行数值全为 0 → ≥1 failure
                         (与原脚本相同, 应正好 10,739 条 = 10.1%)
  - 补充 (+BandCenter) : 主表口径 ∪ BandCenter 有 NaN 或 inf
                         (BandCenter 失败的样本在 L2 里同样被插补)

v2 改动:
  1. 数据源换成论文主结果的预测: 官方 5 折、run0709、截断后的 L1/L2 (XGB 与 RF)
  2. 主表口径不变; 另算加上 BandCenter 的补充版本
  3. 输出列与论文表一致: Model, Subset, n, Level-1 MAE, Level-2 MAE, Level-2 − Level-1, Relative change
     (Relative change = (Level-2 − Level-1) / Level-1)
  4. 核对: 主表失败/完整条数; 两种口径下两子集加权合回的全体 MAE 都要对上 Table 1;
     补充版本的结论 (XGB 两子集都改善, RF 两子集都没有改善) 是否与主表一致

不需要重训。用法 (在 notebooks 目录下):
  python subset_robustness_check_v2.py
  (路径都有默认值, 也可以用 --cache-dir / --v1-pattern / --v2-pattern 覆盖)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TASK_NAME = "matbench_mp_gap"
MODELS = ["xgb", "rf"]
FOLDS = [0, 1, 2, 3, 4]

STRUCT_FEATURIZERS = [
    "DensityFeatures", "GlobalSymmetryFeatures", "StructuralHeterogeneity",
    "ChemicalOrdering", "Dimensionality", "SiteStatsFingerprint",
]

EXPECTED_FAILURE  = 10_739
EXPECTED_COMPLETE = 95_374
# 论文主结果的全体 MAE (eV), 用于核对两子集加权合回
REFERENCE_MAE = {("xgb", "v1"): 0.3380, ("xgb", "v2"): 0.2855,
                 ("rf",  "v1"): 0.3281, ("rf",  "v2"): 0.3379}
TOL = 5e-4


def parse_args():
    ap = argparse.ArgumentParser(description="Level-1 vs Level-2 error on complete / ≥1-failure subsets.")
    ap.add_argument("--cache-dir", type=Path, default=Path("../matbench_cache"))
    ap.add_argument("--v1-pattern", default="../outputs_v1_run0709/predictions_{model}/pred_fold_{fold}.npy")
    ap.add_argument("--v2-pattern", default="../matbench_outputs_v2_run0709/predictions_{model}/pred_fold_{fold}.npy")
    return ap.parse_args()


def _load_cache(cache_dir, fold, feat, n_rows):
    p = cache_dir / f"fold_{fold}_test_{feat}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"缺缓存文件 {p}")
    df = pd.read_pickle(p).reset_index(drop=True)
    if len(df) != n_rows:
        raise ValueError(f"{p.name}: 行数 {len(df)} ≠ 官方 test 数 {n_rows}")
    return df


def failure_masks(fold: int, cache_dir: Path, n_rows: int):
    """返回 (主表口径 mask, BandCenter 失败 mask, 各结构特征器 mask 字典)"""
    per_feat = {}
    for feat in STRUCT_FEATURIZERS:                      # 主表: 与原脚本完全相同
        df = _load_cache(cache_dir, fold, feat, n_rows)
        num = df.select_dtypes(include=[np.number])
        nan_any = df.isna().any(axis=1)
        zero_all = (num == 0).all(axis=1) & ~nan_any
        per_feat[feat] = (nan_any | zero_all).to_numpy()
    main_mask = np.any(np.column_stack(list(per_feat.values())), axis=1)

    bc = _load_cache(cache_dir, fold, "BandCenter", n_rows)   # 补充: BandCenter 的 NaN 或 inf
    bc_num = bc.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    bc_mask = bc.isna().any(axis=1).to_numpy() | np.isinf(bc_num).any(axis=1)
    return main_mask, bc_mask, per_feat


def subset_table(preds, y, bad):
    rows = []
    for model in MODELS:
        p1 = np.concatenate(preds[(model, "v1")])
        p2 = np.concatenate(preds[(model, "v2")])
        for name, mask in [("Complete", ~bad), ("≥1 failure", bad)]:
            m1 = float(np.mean(np.abs(p1[mask] - y[mask])))
            m2 = float(np.mean(np.abs(p2[mask] - y[mask])))
            rows.append({"Model": model.upper(), "Subset": name, "n": int(mask.sum()),
                         "Level-1 MAE (eV)": m1, "Level-2 MAE (eV)": m2,
                         "Level-2 − Level-1 (eV)": m2 - m1, "Relative change": (m2 - m1) / m1})
    return pd.DataFrame(rows)


def print_table(tab):
    show = tab.copy()
    for c in ["Level-1 MAE (eV)", "Level-2 MAE (eV)"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    show["Level-2 − Level-1 (eV)"] = tab["Level-2 − Level-1 (eV)"].map(lambda v: f"{v:+.4f}")
    show["Relative change"] = tab["Relative change"].map(lambda v: f"{v:+.1%}")
    show["n"] = tab["n"].map(lambda v: f"{v:,}")
    print(show.to_string(index=False))


def recombine_check(tab, fold_mae, label):
    """两子集按样本数加权合回的全体 MAE 必须对上 Table 1"""
    print(f"\n核对 [{label}] 两子集加权合回的全体 MAE vs Table 1 (允许差 {TOL}):")
    ok_all = True
    for model in MODELS:
        sub = tab[tab["Model"] == model.upper()].set_index("Subset")
        for ver, col in [("v1", "Level-1 MAE (eV)"), ("v2", "Level-2 MAE (eV)")]:
            pooled = float((sub[col] * sub["n"]).sum() / sub["n"].sum())
            fmean = float(np.mean(fold_mae[(model, ver)]))
            ref = REFERENCE_MAE[(model, ver)]
            ok = abs(pooled - ref) <= TOL
            ok_all &= ok
            lvl = "L1" if ver == "v1" else "L2"
            print(f"  {lvl} {model.upper():<3}: 加权合回 {pooled:.4f} | 五折平均 {fmean:.4f} | "
                  f"Table 1 {ref:.4f}  {'✅' if ok else '❌'}")
    return ok_all


def conclusion(tab):
    """每个 (模型, 子集) 是否改善 (L2 < L1)"""
    return {(r["Model"], r["Subset"]): r["Level-2 − Level-1 (eV)"] < 0 for _, r in tab.iterrows()}


def main():
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False)
    task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
    task.load()
    target_col = task.metadata["target"]

    ys, mains, bcs = [], [], []
    feat_counts = {f: 0 for f in STRUCT_FEATURIZERS}
    preds = {(m, v): [] for m in MODELS for v in ("v1", "v2")}
    fold_mae = {(m, v): [] for m in MODELS for v in ("v1", "v2")}

    for fold in FOLDS:
        y = task.get_test_data(fold, as_type="df", include_target=True)[target_col]
        y = y.reset_index(drop=True).to_numpy(dtype=np.float64)
        main_mask, bc_mask, per_feat = failure_masks(fold, cache_dir, len(y))
        for f, mk in per_feat.items():
            feat_counts[f] += int(mk.sum())
        ys.append(y); mains.append(main_mask); bcs.append(bc_mask)

        for model in MODELS:
            for ver, pattern in [("v1", args.v1_pattern), ("v2", args.v2_pattern)]:
                p = Path(pattern.format(fold=fold, model=model))
                if not p.exists():
                    raise FileNotFoundError(f"找不到预测文件 {p}")
                pred = np.load(p, allow_pickle=False).astype(np.float64)
                if len(pred) != len(y):
                    raise ValueError(f"{p}: 预测长度 {len(pred)} ≠ 标签 {len(y)}")
                if (pred < 0).any():
                    raise ValueError(f"{p}: 有负预测, 这不是截断后的预测")
                preds[(model, ver)].append(pred)
                fold_mae[(model, ver)].append(float(np.mean(np.abs(pred - y))))

    y, bad_main, bc = map(np.concatenate, (ys, mains, bcs))
    bad_ext = bad_main | bc
    n_total = len(y)

    # ---- 1. 两个并集的条数 ----
    n_main = int(bad_main.sum())
    print("\n失败条数:")
    print(f"  主表 (6 个结构特征器, NaN 或整行为 0): {n_main} ({n_main / n_total:.1%})  "
          f"核对 {EXPECTED_FAILURE} {'✅' if n_main == EXPECTED_FAILURE else '❌'}")
    print(f"  补充 (主表 ∪ BandCenter NaN/inf):       {int(bad_ext.sum())} ({bad_ext.mean():.1%})")
    print(f"    其中 BandCenter 失败 {int(bc.sum())} 条, 不在主表失败组里的 {int((bc & ~bad_main).sum())} 条")
    print("  各结构特征器失败条数 (一条可在多个特征器上失败): " +
          ", ".join(f"{f} {c}" for f, c in feat_counts.items()))
    if n_main != EXPECTED_FAILURE or n_total - n_main != EXPECTED_COMPLETE:
        raise ValueError("主表失败/完整条数与核对值不符, 先查失败标记")

    # ---- 2. 两种口径各算一遍 ----
    tab_main = subset_table(preds, y, bad_main)
    tab_ext = subset_table(preds, y, bad_ext)

    print("\n" + "=" * 100 + "\n主表 (结构特征器口径)\n" + "=" * 100)
    print_table(tab_main)
    ok_main = recombine_check(tab_main, fold_mae, "主表")

    print("\n" + "=" * 100 + "\n补充 (含 BandCenter) — 仅供核对, 论文只报失败组条数和结论\n" + "=" * 100)
    print_table(tab_ext)
    ok_ext = recombine_check(tab_ext, fold_mae, "补充")

    # ---- 3. 结论是否一致: XGB 两子集都改善, RF 两子集都没有改善 ----
    expected = {("XGB", "Complete"): True, ("XGB", "≥1 failure"): True,
                ("RF", "Complete"): False, ("RF", "≥1 failure"): False}
    c_main, c_ext = conclusion(tab_main), conclusion(tab_ext)
    print("\n结论核对 (改善 = Level-2 MAE < Level-1 MAE):")
    for key, exp in expected.items():
        print(f"  {key[0]:<3} {key[1]:<11}: 预期 {'改善' if exp else '不改善'} | "
              f"主表 {'改善' if c_main[key] else '不改善'} | 补充 {'改善' if c_ext[key] else '不改善'}  "
              f"{'✅' if c_main[key] == exp and c_ext[key] == exp else '❌'}")
    same = all(c_main[k] == exp and c_ext[k] == exp for k, exp in expected.items())

    n_ext = int(bad_ext.sum())
    print("\n表下补充句 (可直接用):")
    if same:
        print(f'  "Counting BandCenter failures as well gives {n_ext:,} entries with ≥1 failure; '
              f'the conclusion is unchanged."')
    else:
        print("  ⚠️ 补充口径下结论有变化, 不能直接写 the conclusion is unchanged, 先看上面两张表")

    print("\n全部核对通过 ✅" if (ok_main and ok_ext and same) else "\n有核对不通过 ❌")


if __name__ == "__main__":
    main()
