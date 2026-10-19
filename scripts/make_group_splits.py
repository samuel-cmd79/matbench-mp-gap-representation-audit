"""
生成按化学式分组的外层 / 内层划分 (L1 和 L2 共用, 只生成一次)

外层: GroupKFold(n_splits=5), 按 reduced formula 分组, 默认不打乱, 行顺序 = matbench 数据集原始顺序
内层: 对每个外层训练折, GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
      注意 test_size 是"组"的比例, 实际验证样本占比会偏离 20%, 已记录在 split_meta.json

输出 (SPLIT_DIR 下):
  folds.csv         每条一行: id, formula, formula_count_global, y_true, group_fold, official_fold, official_pos
  inner_splits.csv  每个 (外层折, 训练条目) 一行: outer_fold, id, role (fit / val)
  split_meta.json   版本号、规则、每折样本数/组数/验证占比、三项检查结果

用法: 在 notebooks 目录下运行  python make_group_splits.py
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# ==================== 配置区 ====================
TASK_NAME          = "matbench_mp_gap"
N_SPLITS           = 5
INNER_TEST_SIZE    = 0.2
INNER_RANDOM_STATE = 42
EXPECTED_N         = 106_113
SPLIT_DIR          = Path("../groupcv_splits")
OVERWRITE          = False     # 划分文件已存在时是否覆盖 (默认不覆盖, 防止跑到一半换了划分)
# ================================================


def _versions():
    import pymatgen.core
    v = {"python": platform.python_version(), "numpy": np.__version__,
         "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
         "pymatgen": getattr(pymatgen.core, "__version__", "unknown")}
    try:
        import importlib.metadata as md
        v["matbench"] = md.version("matbench")
    except Exception:
        v["matbench"] = "unknown"
    return v


def main():
    out_files = [SPLIT_DIR / n for n in ("folds.csv", "inner_splits.csv", "split_meta.json")]
    if any(p.exists() for p in out_files) and not OVERWRITE:
        raise FileExistsError(f"{SPLIT_DIR} 里已有划分文件; 确实要重新生成请设 OVERWRITE = True")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False)
    task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
    task.load()
    tcol = task.metadata["target"].strip()

    df = task.df.rename(columns=lambda x: x.strip())
    ids = np.array([str(i) for i in df.index], dtype=object)
    n = len(ids)
    assert n == EXPECTED_N, f"数据集条数 {n} ≠ {EXPECTED_N}"
    assert len(set(ids)) == n, "数据集 id 有重复"
    y = df[tcol].to_numpy(dtype=np.float64)

    print("计算 reduced formula ...")
    formula = np.array([s.composition.reduced_formula for s in df["structure"]], dtype=object)
    counts = pd.Series(formula).value_counts()
    formula_count = pd.Series(formula).map(counts).to_numpy()

    # ---- 官方 5 折: 每个 id 属于哪个测试折、在该折中的位置 (L1/L2 缓存行号按这个顺序) ----
    off_fold = pd.Series(-1, index=ids)
    off_pos = pd.Series(-1, index=ids)
    for fold in task.folds:
        tdf = task.get_test_data(fold, as_type="df", include_target=True).rename(columns=lambda x: x.strip())
        tids = [str(i) for i in tdf.index]
        assert set(tids) <= set(ids), f"官方 fold {fold} 有不在数据集里的 id"
        assert (off_fold.loc[tids] == -1).all(), f"官方 fold {fold} 的 id 在别的测试折里出现过"
        assert np.array_equal(tdf[tcol].to_numpy(dtype=np.float64), pd.Series(y, index=ids).loc[tids].to_numpy()), \
            f"官方 fold {fold} 的真值与 task.df 不一致"
        off_fold.loc[tids] = int(fold)
        off_pos.loc[tids] = np.arange(len(tids))
    assert (off_fold >= 0).all(), "有条目不在任何官方测试折里"

    # ---- 外层: GroupKFold ----
    group_fold = np.full(n, -1)
    for k, (_, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(np.zeros(n), groups=formula)):
        group_fold[te] = k
    assert (group_fold >= 0).all()

    # ---- 内层: 每个外层训练折一次 GroupShuffleSplit ----
    inner_rows, per_fold = [], []
    for k in range(N_SPLITS):
        tr = np.where(group_fold != k)[0]
        te = np.where(group_fold == k)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=INNER_TEST_SIZE, random_state=INNER_RANDOM_STATE)
        fit_rel, val_rel = next(gss.split(np.zeros(len(tr)), groups=formula[tr]))
        fit, val = tr[fit_rel], tr[val_rel]

        # 检查
        ov_outer = set(formula[tr]) & set(formula[te])
        ov_inner = set(formula[fit]) & set(formula[val])
        assert not ov_outer, f"外层 fold {k}: 训练/测试化学式交集 {len(ov_outer)}"
        assert not ov_inner, f"外层 fold {k}: 内层 fit/val 化学式交集 {len(ov_inner)}"
        assert len(fit) + len(val) == len(tr) and not (set(fit) & set(val))

        role = np.empty(len(tr), dtype=object)
        role[fit_rel], role[val_rel] = "fit", "val"
        inner_rows.append(pd.DataFrame({"outer_fold": k, "id": ids[tr], "role": role}))
        per_fold.append({
            "fold": k, "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_fit": int(len(fit)), "n_val": int(len(val)),
            "val_fraction_of_outer_train": round(len(val) / len(tr), 6),
            "n_groups_train": int(len(set(formula[tr]))), "n_groups_test": int(len(set(formula[te]))),
            "n_groups_fit": int(len(set(formula[fit]))), "n_groups_val": int(len(set(formula[val]))),
            "formula_overlap_train_test": len(ov_outer), "formula_overlap_fit_val": len(ov_inner),
        })

    folds = pd.DataFrame({
        "id": ids, "formula": formula, "formula_count_global": formula_count, "y_true": y,
        "group_fold": group_fold, "official_fold": off_fold.to_numpy(), "official_pos": off_pos.to_numpy(),
    })
    inner = pd.concat(inner_rows, ignore_index=True)
    assert len(folds) == EXPECTED_N and folds["group_fold"].value_counts().sum() == EXPECTED_N

    meta = {
        "task": TASK_NAME, "n_total": int(n), "n_unique_formulas": int(len(counts)),
        "n_singleton_formulas": int((counts == 1).sum()),
        "n_entries_with_repeated_formula": int((formula_count > 1).sum()),
        "grouping": "pymatgen structure.composition.reduced_formula",
        "row_order": "matbench task.df original order",
        "outer": {"method": "GroupKFold", "n_splits": N_SPLITS, "shuffle": False},
        "inner": {"method": "GroupShuffleSplit", "n_splits": 1, "test_size": INNER_TEST_SIZE,
                  "test_size_unit": "fraction of groups (formulas), not samples",
                  "random_state": INNER_RANDOM_STATE},
        "checks": {"total_equals_expected": True, "all_outer_formula_overlap_zero": True,
                   "all_inner_formula_overlap_zero": True},
        "per_fold": per_fold, "versions": _versions(),
    }

    folds.to_csv(SPLIT_DIR / "folds.csv", index=False)
    inner.to_csv(SPLIT_DIR / "inner_splits.csv", index=False)
    (SPLIT_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ 已写出 {SPLIT_DIR.resolve()}")
    print(f"   总数 {n}, 化学式 {len(counts)} 个, 其中只出现一次的 {(counts == 1).sum()} 个")
    print(pd.DataFrame(per_fold)[["fold", "n_train", "n_test", "n_fit", "n_val",
                                  "val_fraction_of_outer_train", "n_groups_test",
                                  "formula_overlap_train_test", "formula_overlap_fit_val"]].to_string(index=False))


if __name__ == "__main__":
    main()
