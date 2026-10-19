"""
按 (配置, 种子, 折) 逐行汇总 MAE / RMSE / best_iteration。

- MAE、RMSE 用截断后的预测 pred_fold_{k}.npy 计算（与论文一致）
- 真值来自 matbench 官方测试集，顺序与预测文件一致
- best_iteration 查找顺序:
    1) 输出目录里的 xgb_training_info_fold_{k+1}.json   (v1 有, v2/消融没有)
    2) LOGS 里指定的、已执行并保存了输出的 .ipynb，从打印日志里解析
    3) 都没有 → 留空, 并在最后列出缺失
- 输出: per_fold_metrics.csv (每个配置×种子×折一行), paired_deltas.csv (删除 − 完整, 同种子同折配对)
  所有浮点保留 6 位小数; 差值用未四舍五入的原值相减后再保留 6 位。

用法: 在 notebooks 目录下运行  python collect_fold_metrics.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ==================== 配置区: 只需要改这里 ====================
# (配置名, 种子) -> 输出目录。目录名请按你实际跑的改; 不存在的会跳过并提示。
RUNS = {
    ("L2_full",            42):  "../matbench_outputs_v2_run0709",
    ("L2_full",            101): "../matbench_outputs_v2_seed101",
    ("L2_full",            202): "../matbench_outputs_v2_seed202",
    ("L2_no_coordination", 42):  "../matbench_outputs_v2_ablation_coordination",
    ("L2_no_coordination", 101): "../matbench_outputs_v2_ablation_coordination_seed101",
    ("L2_no_coordination", 202): "../matbench_outputs_v2_ablation_coordination_seed202",
    ("L2_no_symmetry",     42):  "../matbench_outputs_v2_ablation_symmetry",
    ("L2_no_symmetry",     101): "../matbench_outputs_v2_ablation_symmetry_seed101",
    ("L2_no_symmetry",     202): "../matbench_outputs_v2_ablation_symmetry_seed202",
    # L1 (v1, 只有成分特征); 跑了其他种子就照样加行
    ("L1",                 42):  "../outputs_v1_run0709",
    ("L1",                 101): "../outputs_v1_seed101",
    ("L1",                 202): "../outputs_v1_seed202",
}
MODEL = "xgb"

# 可选: (配置名, 种子) -> 已执行并保存输出的 notebook, 用来解析 best_iteration
LOGS = {
    # ("L2_full", 101): "mp_gap_baseline_v2_seed101.ipynb",
}

BASELINE = "L2_full"     # 配对相减时的基准配置
OUT_DIR  = Path("../results/multiseed")      # 结果 csv 存放位置
TASK_NAME = "matbench_mp_gap"
# ==============================================================

LOG_PATTERNS = [
    re.compile(r"\[XGBoost Fold (\d+)\] 完成: 最优轮数=(\d+)"),   # v2 / 消融
    re.compile(r"\[XGB Fold (\d+)\] best_iteration = (\d+)"),     # v1
]


def load_truth():
    """返回 {fold: y_true}，顺序与 get_test_data 一致（预测文件就是按这个顺序写的）"""
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False)
    task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
    task.load()
    target_col = task.metadata["target"].strip()
    truth = {}
    for fold in task.folds:
        df = task.get_test_data(fold, as_type="df", include_target=True)
        df = df.rename(columns=lambda x: x.strip())
        truth[int(fold)] = df[target_col].to_numpy(dtype=np.float64)
    return truth


def best_its_from_log(nb_path):
    """从 notebook 保存的输出里解析 {fold_idx(0-based): best_iteration}"""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    text = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            t = out.get("text", "")
            text.append("".join(t) if isinstance(t, list) else t)
    text = "\n".join(text)
    found = {}
    for pat in LOG_PATTERNS:
        for m in pat.finditer(text):
            k, it = int(m.group(1)) - 1, int(m.group(2))
            if k in found and found[k] != it:
                raise ValueError(f"{nb_path}: fold {k} 出现两个不同的 best_iteration {found[k]} / {it}")
            found[k] = it
    return found


def best_iteration(run_dir, fold, log_its):
    j = run_dir / f"xgb_training_info_fold_{fold + 1}.json"
    if j.exists():
        return int(json.loads(j.read_text(encoding="utf-8"))["best_iteration"]), "json"
    if fold in log_its:
        return log_its[fold], "log"
    return None, "missing"


def collect(truth):
    rows, missing_dirs = [], []
    for (config, seed), d in RUNS.items():
        run_dir = Path(d).resolve()
        pred_dir = run_dir / f"predictions_{MODEL}"
        if not pred_dir.exists():
            missing_dirs.append((config, seed, str(pred_dir)))
            continue
        log_its = best_its_from_log(LOGS[(config, seed)]) if (config, seed) in LOGS else {}

        for fold, y_true in truth.items():
            f = pred_dir / f"pred_fold_{fold}.npy"
            if not f.exists():
                raise FileNotFoundError(f"缺少预测文件: {f}")
            y_pred = np.load(f, allow_pickle=False).astype(np.float64)
            if y_pred.shape != y_true.shape:
                raise ValueError(f"{f}: 长度 {y_pred.shape} 与真值 {y_true.shape} 不一致")
            if not np.isfinite(y_pred).all():
                raise ValueError(f"{f}: 含 NaN/Inf")
            if (y_pred < 0).any():
                raise ValueError(f"{f}: 存在负值, 这不是截断后的预测")

            err = y_pred - y_true
            it, src = best_iteration(run_dir, fold, log_its)
            rows.append({
                "config": config, "seed": seed, "fold": fold, "n": y_true.size,
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "best_iteration": it, "best_iteration_source": src,
                "source_dir": d,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["best_iteration"] = df["best_iteration"].astype("Int64")
    return df, missing_dirs


def paired_deltas(df):
    base = df[df.config == BASELINE].set_index(["seed", "fold"])
    out = []
    for config in df.config.unique():
        if config in (BASELINE, "L1"):
            continue
        abl = df[df.config == config].set_index(["seed", "fold"])
        common = abl.index.intersection(base.index)
        for key in common:
            out.append({
                "config": config, "seed": key[0], "fold": key[1],
                "dMAE":  abl.at[key, "MAE"]  - base.at[key, "MAE"],
                "dRMSE": abl.at[key, "RMSE"] - base.at[key, "RMSE"],
            })
    return pd.DataFrame(out)


def main():
    truth = load_truth()
    print(f"真值: {len(truth)} 折, 共 {sum(v.size for v in truth.values())} 条")

    df, missing_dirs = collect(truth)
    df = df.sort_values(["config", "seed", "fold"]).reset_index(drop=True)
    p1 = OUT_DIR / "per_fold_metrics.csv"
    df.to_csv(p1, index=False, float_format="%.6f")
    print(f"\n✅ 已写出 {p1}  ({len(df)} 行)")

    dd = paired_deltas(df)
    if not dd.empty:
        dd = dd.sort_values(["config", "seed", "fold"]).reset_index(drop=True)
        p2 = OUT_DIR / "paired_deltas.csv"
        dd.to_csv(p2, index=False, float_format="%.6f")
        print(f"✅ 已写出 {p2}  ({len(dd)} 行, 删除 − {BASELINE}, 同种子同折)")

    # 五折平均, 仅供和论文数字核对
    with pd.option_context("display.float_format", "{:.6f}".format):
        print("\n五折平均 (核对用; seed 42 的 L2_full 应等于论文/官方 MAE):")
        print(df.groupby(["config", "seed"])[["MAE", "RMSE"]].mean())

    if missing_dirs:
        print("\n⚠️ 以下目录不存在, 已跳过:")
        for c, s, p in missing_dirs:
            print(f"   {c} seed {s}: {p}")
    miss_it = df[df.best_iteration_source == "missing"]
    if len(miss_it):
        print(f"\n⚠️ {len(miss_it)} 行缺 best_iteration:")
        print(miss_it.groupby(["config", "seed"]).fold.apply(list).to_string())


if __name__ == "__main__":
    main()
