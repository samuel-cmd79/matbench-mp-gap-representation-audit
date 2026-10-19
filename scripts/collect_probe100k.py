"""
汇总 100000 轮上限 (probe50k) 下 L1 / L2 完整 / 删配位 / 删对称性 的逐折结果。

每个 (配置, 折) 一行: MAE, RMSE (截断后预测), best_iteration, 实际轮数, 是否早停。
另外输出: 五折平均, 以及 删除 − L2 完整 的同折配对差值。
浮点保留 6 位小数。

用法: 在 notebooks 目录下运行  python collect_probe50k.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ==================== 配置区 ====================
CAP = 100000          # 本次的轮数上限
ES_ROUNDS = 200      # 早停容忍度
RUNS = {
    # 配置名: (目录, json 格式, 官方分数文件)
    "L1":                 ("../outputs_v1_run0709_replay_probe50k",                "v1", "metrics_xgb.txt"),
    "L2_full":            ("../matbench_outputs_v2_run0709_replay_probe50k",       "v2", "scores_xgb.txt"),
    "L2_no_coordination": ("../matbench_outputs_v2_ablation_coordination_probe50k", "v2", "scores_xgb.txt"),
    "L2_no_symmetry":     ("../matbench_outputs_v2_ablation_symmetry_probe50k",     "v2", "scores_xgb.txt"),
}
BASELINE = "L2_full"
CHECK_TOL = 1e-9     # 与官方五折平均的允许误差 (eV)
MODEL = "xgb"
OUT_DIR = Path("../results/round_cap")
TASK_NAME = "matbench_mp_gap"
# ================================================


def load_truth():
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False)
    task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
    task.load()
    tcol = task.metadata["target"].strip()
    truth = {}
    for fold in task.folds:
        df = task.get_test_data(fold, as_type="df", include_target=True)
        df = df.rename(columns=lambda x: x.strip())
        truth[int(fold)] = df[tcol].to_numpy(dtype=np.float64)
    return truth


def read_official_mean(path, metric):
    """从 scores/metrics txt 里解析官方五折平均, 如 'mae': {..., 'mean': 0.123, ...}; 解析不到返回 None"""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"['\"]" + metric + r"['\"]\s*:\s*\{[^{}]*?['\"]mean['\"]\s*:\s*"
                  r"(?:np\.float\d*\()?\s*([-+0-9.eE]+)", text)
    return float(m.group(1)) if m else None


def official_check(df):
    """五折都在的配置: 脚本算的五折平均 vs 官方 txt 里的 mean"""
    lines = []
    for config, (d, _, score_file) in RUNS.items():
        sub = df[df.config == config]
        f = Path(d).resolve() / score_file
        if len(sub) != 5:
            lines.append(f"   {config}: 只有 {len(sub)} 折, 跳过核对")
            continue
        if not f.exists():
            lines.append(f"   {config}: 没有 {score_file}, 跳过核对")
            continue
        for metric, col in [("mae", "MAE"), ("rmse", "RMSE")]:
            off = read_official_mean(f, metric)
            if off is None:
                lines.append(f"   {config} {col}: {score_file} 里解析不到 {metric}.mean, 请手动核对")
                continue
            mine = float(sub[col].mean())
            ok = abs(mine - off) <= CHECK_TOL
            lines.append(f"   {config} {col}: 脚本 {mine:.6f} vs 官方 {off:.6f}  "
                         f"差 {mine - off:+.2e}  {'✅' if ok else '❌'}")
    return lines


def read_info(run_dir, kind, fold):
    """返回 (best_iteration, n_rounds_run, json 文件名)；找不到返回 (None, None, None)"""
    if kind == "v1":
        f = run_dir / f"xgb_training_info_fold_{fold + 1}.json"          # v1: 编号从 1 开始
        if not f.exists():
            return None, None, None
        d = json.loads(f.read_text(encoding="utf-8"))
        return int(d["best_iteration"]), int(d["total_rounds_evaluated"]), f.name
    files = [p for p in run_dir.glob(f"xgb_training_info_*_fold_{fold}.json")
             if "testmode" not in p.name]                                  # v2/消融: 编号从 0 开始
    if not files:
        return None, None, None
    if len(files) > 1:
        raise ValueError(f"{run_dir} fold {fold} 有多个训练信息文件: {[p.name for p in files]}")
    d = json.loads(files[0].read_text(encoding="utf-8"))
    recorded_cap = d.get("num_boost_round_max")

    # 唯一例外：L2_full fold 0 使用旧的 50000 轮上限
    expected_cap = (
        50000
        if run_dir.name == "matbench_outputs_v2_run0709_replay_probe50k"
        and fold == 0
        else CAP
    )

    if recorded_cap != expected_cap:
        raise ValueError(
            f"{files[0]}: 上限是 {recorded_cap}, 预期是 {expected_cap}"
        )
    return int(d["best_iteration"]), int(d["n_rounds_run"]), files[0].name


def main():
    truth = load_truth()
    rows, problems = [], []

    for config, (d, kind, _) in RUNS.items():
        run_dir = Path(d).resolve()
        pred_dir = run_dir / f"predictions_{MODEL}"
        for fold, y in truth.items():
            pf = pred_dir / f"pred_fold_{fold}.npy"
            if not pf.exists():
                problems.append(f"{config} fold {fold}: 缺预测文件 {pf}")
                continue
            p = np.load(pf, allow_pickle=False).astype(np.float64)
            if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any():
                raise ValueError(f"{pf}: 形状不符 / 含 NaN / 含负值")
            e = p - y

            best_it, n_run, info_name = read_info(run_dir, kind, fold)
            if best_it is None:
                problems.append(f"{config} fold {fold}: 缺训练信息 json")
                early = None
            else:
                early = n_run < CAP
                if not early:
                    problems.append(f"{config} fold {fold}: 跑满 {CAP} 轮, 上限仍然撞到了")
                elif n_run != best_it + 1 + ES_ROUNDS:
                    problems.append(f"{config} fold {fold}: 实际轮数 {n_run} ≠ best_iteration+1+{ES_ROUNDS}")

            rows.append({
                "config": config, "fold": fold, "n": y.size,
                "MAE": float(np.mean(np.abs(e))),
                "RMSE": float(np.sqrt(np.mean(e ** 2))),
                "best_iteration": best_it, "n_rounds_run": n_run, "early_stopped": early,
                "info_file": info_name,
            })

    df = pd.DataFrame(rows)
    df["best_iteration"] = df["best_iteration"].astype("Int64")
    df["n_rounds_run"] = df["n_rounds_run"].astype("Int64")
    order = {c: i for i, c in enumerate(RUNS)}
    df = df.sort_values(["config", "fold"], key=lambda s: s.map(order) if s.name == "config" else s)
    df = df.reset_index(drop=True)
    df.to_csv(OUT_DIR / "probe50k_per_fold.csv", index=False, float_format="%.6f")

    # 配对差值: 删除 − 完整, 同折 (用原值相减)
    base = df[df.config == BASELINE].set_index("fold")
    deltas = []
    for config in RUNS:
        if config in (BASELINE, "L1"):
            continue
        abl = df[df.config == config].set_index("fold")
        for fold in abl.index.intersection(base.index):
            deltas.append({"config": config, "fold": fold,
                           "dMAE": abl.at[fold, "MAE"] - base.at[fold, "MAE"],
                           "dRMSE": abl.at[fold, "RMSE"] - base.at[fold, "RMSE"]})
    dd = pd.DataFrame(deltas)
    if not dd.empty:
        dd.to_csv(OUT_DIR / "probe50k_paired_deltas.csv", index=False, float_format="%.6f")

    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.width", 200, "display.max_columns", 20):
        print("逐折结果:")
        print(df.drop(columns=["info_file"]).to_string(index=False))
        print("\n五折平均:")
        print(df.groupby("config", sort=False)[["MAE", "RMSE"]].mean()
                .assign(n_folds=df.groupby("config", sort=False).size()))
        if not dd.empty:
            print(f"\n配对差值 (删除 − {BASELINE}, 同折):")
            print(dd.to_string(index=False))
            print("\n配对差值五折平均:")
            print(dd.groupby("config", sort=False)[["dMAE", "dRMSE"]].mean())

    print("\n与官方五折平均核对 (一致才说明真值和预测对齐):")
    for line in official_check(df):
        print(line)

    print("\n✅ 已写出 probe50k_per_fold.csv" + (", probe50k_paired_deltas.csv" if not dd.empty else ""))
    if problems:
        print("\n⚠️ 需要注意:")
        for p in problems:
            print("   " + p)


if __name__ == "__main__":
    main()
