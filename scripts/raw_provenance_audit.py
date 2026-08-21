#!/usr/bin/env python3
"""
raw_provenance_audit.py  --  raw 预测来源审计

目的：给论文里每一个 raw 统计量盖一个来源戳，并用可复算的数字证明这个戳成立。

三个 Level 的证据等级不同：
  L1  归档 raw + 冻结 clipped  -> 正值子集是直接验证；负值子集靠模型同一性推断
  L2  归档 raw 用的是旧插补策略 -> 与冻结模型不等价，只能作诊断
  L3  raw / clipped 同一次冻结运行 -> 直接验证，可正式使用

本脚本只读文件，不训练、不改写任何原始数据。
输出：终端摘要 + raw_provenance_audit.md
"""

import json
import numpy as np
from pathlib import Path

# ============================================================
# 配置：路径按你的实际目录改；{k} 会被 0..4 替换
# ============================================================
FOLDS = range(5)

P_L1_ARCHIVED_RAW = "../matbench_outputs/v1_predictions_xgb/pred_fold_{k}.npy"
P_L1_FROZEN_CLIP  = "../outputs_v1_run0709/predictions_xgb/pred_fold_{k}.npy"
P_L2_ARCHIVED_RAW = "../matbench_outputs/v2_predictions_xgb/pred_fold_{k}.npy"
P_L2_FROZEN_CLIP  = "../matbench_outputs_v2_run0709/predictions_xgb/pred_fold_{k}.npy"
P_L3_FROZEN_RAW   = "../results_v4/fold_{k}/test_preds.npz"
P_L3_FROZEN_CLIP  = "../results_v4/fold_{k}/test_preds_clipped.npz"

# 真实标签。没有就设成 None，脚本会跳过所有需要标签的统计（对账部分仍然会跑）。
# 生成方法见文件末尾注释。
P_LABELS = "labels_by_fold.npz"

# 训练日志。json 是 1 起、npy 是 0 起，这里显式写死映射关系，别用 zip。
P_L1_FROZEN_LOG   = "../outputs_v1_run0709/xgb_training_info_fold_{k1}.json"   # k1 = k + 1
P_L1_ARCHIVED_LOG = "../outputs/xgb_training_info_fold_{k1}.json"   # 归档运行如果没存日志就保持 None

N_EXPECTED = 106113
N_ZERO_EXPECTED = 46151
TOLS = [0.0, 1e-15, 1e-12, 1e-9]

OUT_MD = "raw_provenance_audit.md"

report = []           # markdown 行
def say(line=""):
    print(line)
    report.append(line)


# ============================================================
# 读取
# ============================================================
def load_npy_folds(pattern):
    """读 5 个 .npy，返回 list of 1-D array。文件缺失直接报错，不猜、不替代。"""
    out = []
    for k in FOLDS:
        p = Path(pattern.format(k=k))
        if not p.exists():
            raise FileNotFoundError(f"缺文件：{p}")
        out.append(np.load(p).astype(np.float64).ravel())
    return out


def load_npz_folds(pattern):
    """读 5 个 .npz（含 ids / preds），返回 (ids_list, preds_list)。"""
    ids, preds = [], []
    for k in FOLDS:
        p = Path(pattern.format(k=k))
        if not p.exists():
            raise FileNotFoundError(f"缺文件：{p}")
        z = np.load(p)
        ids.append(np.asarray(z["ids"]))
        preds.append(np.asarray(z["preds"], dtype=np.float64).ravel())
    return ids, preds


def cat(list_of_arrays):
    return np.concatenate(list_of_arrays)


# ============================================================
# 对账工具
# ============================================================
def reconcile(a, b, label):
    """逐样本比较两个数组，返回统计字典。不使用 np.allclose——它默认容差 1e-8，会藏住差异。"""
    d = np.abs(a - b)
    r = {
        "label": label,
        "n": int(a.size),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
    }
    for t in TOLS:
        key = "exact_unequal" if t == 0.0 else f"unequal_gt_{t:g}"
        r[key] = int((d > t).sum())
    return r


def fmt_reconcile(r):
    lines = [
        f"- **{r['label']}**  (n = {r['n']:,})",
        f"  - 最大绝对差：`{r['max_abs_diff']:.3e}`",
        f"  - 平均绝对差：`{r['mean_abs_diff']:.3e}`",
    ]
    for t in TOLS:
        key = "exact_unequal" if t == 0.0 else f"unequal_gt_{t:g}"
        tag = "精确不等" if t == 0.0 else f"差 > {t:g}"
        lines.append(f"  - {tag}：**{r[key]:,}** 个")
    return lines


def dist_stats(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(med),
        "IQR": float(q3 - q1),
        "q5": float(np.percentile(x, 5)),
        "q95": float(np.percentile(x, 95)),
        "neg_count": int((x < 0).sum()),
        "neg_share": float((x < 0).mean()),
    }


# ============================================================
# 主流程
# ============================================================
def main():
    say("# Raw 预测来源审计报告")
    say()

    # ---------- 0. 读取 ----------
    say("## 0. 读取与对齐检查")
    say()

    l1_raw_f  = load_npy_folds(P_L1_ARCHIVED_RAW)
    l1_clip_f = load_npy_folds(P_L1_FROZEN_CLIP)
    l2_raw_f  = load_npy_folds(P_L2_ARCHIVED_RAW)
    l2_clip_f = load_npy_folds(P_L2_FROZEN_CLIP)
    l3_ids_f, l3_raw_f  = load_npz_folds(P_L3_FROZEN_RAW)
    l3_ids2_f, l3_clip_f = load_npz_folds(P_L3_FROZEN_CLIP)

    # 每折样本数必须处处一致
    for k in FOLDS:
        sizes = {
            "L1_raw": l1_raw_f[k].size, "L1_clip": l1_clip_f[k].size,
            "L2_raw": l2_raw_f[k].size, "L2_clip": l2_clip_f[k].size,
            "L3_raw": l3_raw_f[k].size, "L3_clip": l3_clip_f[k].size,
        }
        assert len(set(sizes.values())) == 1, f"fold {k} 样本数不一致：{sizes}"
        assert np.array_equal(l3_ids_f[k], l3_ids2_f[k]), f"fold {k} 的 L3 raw/clip ids 不一致"

    fold_sizes = [a.size for a in l1_raw_f]
    say(f"- 每折样本数：{fold_sizes}，合计 **{sum(fold_sizes):,}**"
        f"（预期 {N_EXPECTED:,}：{'OK' if sum(fold_sizes) == N_EXPECTED else '不符'}）")

    # L1/L2 的 .npy 没有 ID，只能按行序对齐。
    # 这里用 L3 的 ids 作为每折的规范 ID —— 前提是所有运行都用了同一套官方折，
    # 且落盘顺序与官方 test 集顺序一致。这个前提由后面 L1 的“裁剪后逐样本零偏差”
    # 反向验证：如果行序错乱，不可能全部相等。
    ids = cat(l3_ids_f)
    say(f"- ID 唯一性：{len(np.unique(ids)):,} / {len(ids):,} "
        f"（{'唯一' if len(np.unique(ids)) == len(ids) else '有重复！'}）")

    l1_raw, l1_clip = cat(l1_raw_f), cat(l1_clip_f)
    l2_raw, l2_clip = cat(l2_raw_f), cat(l2_clip_f)
    l3_raw, l3_clip = cat(l3_raw_f), cat(l3_clip_f)

    for name, arr in [("L1_raw", l1_raw), ("L1_clip", l1_clip), ("L2_raw", l2_raw),
                      ("L2_clip", l2_clip), ("L3_raw", l3_raw), ("L3_clip", l3_clip)]:
        bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
        assert bad == 0, f"{name} 含 {bad} 个 NaN/Inf"
    say("- NaN / Inf 检查：全部通过")

    for name, arr in [("L1_clip", l1_clip), ("L2_clip", l2_clip), ("L3_clip", l3_clip)]:
        n_neg = int((arr < 0).sum())
        say(f"- {name} 负值个数：**{n_neg}**（冻结的裁剪后预测应为 0）")

    # 标签（可选）
    y = None
    if P_LABELS and Path(P_LABELS).exists():
        z = np.load(P_LABELS, allow_pickle=True)
        y_f, ids_f = [], []
        for k in FOLDS:
            y_f.append(np.asarray(z[f"y_{k}"], dtype=np.float64).ravel())
            ids_f.append(np.asarray(z[f"ids_{k}"]))
        # 标签的 ID 必须和 L3 的 ids 逐折一致，否则说明折序对不上
        for k in FOLDS:
            assert np.array_equal(ids_f[k], l3_ids_f[k]), f"fold {k} 标签 ID 与 L3 ids 不一致"
        y = cat(y_f)
        n_zero = int((y == 0).sum())
        say(f"- 真实标签已载入；零带隙样本 **{n_zero:,}**"
            f"（预期 {N_ZERO_EXPECTED:,}：{'OK' if n_zero == N_ZERO_EXPECTED else '不符'}）")
    else:
        say("- 未提供真实标签：跳过所有 MAE / 零带隙统计，只做对账部分")
    say()

    # ---------- 1. Level 1 分层验证 ----------
    say("## 1. Level 1：分层同一性验证")
    say()
    say("把归档 raw 自己裁剪一次，与冻结 clipped 对比。两者若来自同一 booster，应完全相等。")
    say()

    l1_recon_clip = np.maximum(l1_raw, 0.0)
    r_all = reconcile(l1_recon_clip, l1_clip, "全体样本：归档 raw 裁剪后 vs 冻结 clipped")
    for ln in fmt_reconcile(r_all):
        say(ln)
    say()

    n_pos = int((l1_raw > 0).sum())
    n_zer = int((l1_raw == 0).sum())
    n_neg = int((l1_raw < 0).sum())
    say(f"按归档 raw 的符号分层（严格条件，三者互斥且穷尽）：")
    say()
    say(f"| 子集 | 个数 | 占比 |")
    say(f"|---|---:|---:|")
    say(f"| `raw > 0` 严格正 | {n_pos:,} | {n_pos/len(l1_raw):.2%} |")
    say(f"| `raw == 0` 恰好零 | {n_zer:,} | {n_zer/len(l1_raw):.2%} |")
    say(f"| `raw < 0` 负 | {n_neg:,} | {n_neg/len(l1_raw):.2%} |")
    say()
    assert n_pos + n_zer + n_neg == len(l1_raw)

    # 严格正值子集：裁剪是恒等映射，所以这里是货真价实的 raw-vs-raw 直接验证
    pos = l1_raw > 0
    r_pos = reconcile(l1_raw[pos], l1_clip[pos],
                      "严格正值子集：归档 raw vs 冻结 clipped（此处裁剪为恒等，属直接验证）")
    for ln in fmt_reconcile(r_pos):
        say(ln)
    say()
    say("> `raw == 0` 的样本**不**并入直接验证：冻结 raw 可能为负，裁剪后同样是 0，无法分辨。")
    say()

    # 负值子集：只能靠模型同一性推断，这里汇总支撑证据
    say(f"### 负值子集（{n_neg:,} 个）的同一性证据")
    say()
    logs_ok = True
    for k in FOLDS:
        pf = Path(P_L1_FROZEN_LOG.format(k1=k + 1))     # 注意：json 是 1 起
        if not pf.exists():
            say(f"- fold {k}：冻结日志缺失（{pf}）")
            logs_ok = False
            continue
        jf = json.loads(pf.read_text())
        line = (f"- fold {k}（json fold={jf.get('fold')}）："
                f"best_iter={jf.get('best_iteration')}, "
                f"best_val_mae={jf.get('best_validation_mae'):.6f}, "
                f"seed={jf.get('params', {}).get('seed')}")
        if P_L1_ARCHIVED_LOG:
            pa = Path(P_L1_ARCHIVED_LOG.format(k1=k + 1))
            if pa.exists():
                ja = json.loads(pa.read_text())
                same = (ja.get("best_iteration") == jf.get("best_iteration")
                        and abs(ja.get("best_validation_mae", -1)
                                - jf.get("best_validation_mae", -2)) < 1e-12
                        and ja.get("params") == jf.get("params"))
                line += f" | 与归档日志：{'一致' if same else '**不一致**'}"
            else:
                line += " | 归档日志缺失"
        say(line)
    say()
    if not P_L1_ARCHIVED_LOG:
        say("> 归档运行未保存训练日志，因此无法做逐折日志对照。"
            "负值子集的等价性由以下证据支撑：组分特征矩阵无缺失（插补变更为恒等操作）、"
            "严格正值子集逐样本零偏差、配置与 seed 相同。**表述为 identity-inferred，"
            "不得写成 directly bit-verified。**")
    say()

    # ---------- 2. Level 2 反证 ----------
    say("## 2. Level 2：证明其只能作诊断")
    say()
    l2_recon_clip = np.maximum(l2_raw, 0.0)
    r_l2 = reconcile(l2_recon_clip, l2_clip, "归档 raw 裁剪后 vs 冻结 clipped（预期**不**相等）")
    for ln in fmt_reconcile(r_l2):
        say(ln)
    say()
    if r_l2["exact_unequal"] > 0:
        say("> 结论：Level-2 归档 raw 与冻结模型不等价，仅可作为 diagnostic。")
        say("> **不得**用「归档 raw MAE − 冻结 clipped MAE」计算或命名为 clipping gain。")
    else:
        say("> 意外：两者完全相等。若属实，Level 2 也可升级为 frozen-equivalent，需人工复核。")
    say()

    if y is not None:
        # 同一次归档运行内部的 clipping gain 是自洽的，可以报
        gain_l2_within = np.abs(l2_raw - y).mean() - np.abs(l2_recon_clip - y).mean()
        say(f"- 归档运行**内部**的 diagnostic clipping gain："
            f"`{gain_l2_within:.6f}` eV（自洽，可标为 old-run diagnostic）")
        say()

    # ---------- 3. Level 3 同源验证 ----------
    say("## 3. Level 3：同源直接验证")
    say()
    l3_recon_clip = np.maximum(l3_raw, 0.0)
    r_l3 = reconcile(l3_recon_clip, l3_clip, "冻结 raw 裁剪后 vs 冻结 clipped（同一次运行，应精确相等）")
    for ln in fmt_reconcile(r_l3):
        say(ln)
    say()

    if y is not None:
        zero = (y == 0)
        gain_all  = np.abs(l3_raw - y).mean() - np.abs(l3_clip - y).mean()
        gain_zero = np.abs(l3_raw[zero] - y[zero]).mean() - np.abs(l3_clip[zero] - y[zero]).mean()
        say(f"- 全集 clipping gain：`{gain_all:.6f}` eV")
        say(f"- 零带隙子集 clipping gain：`{gain_zero:.6f}` eV "
            f"（稿件写的是 0.0011，{'复现' if abs(gain_zero - 0.0011) < 5e-4 else '**对不上，需查**'}）")
        say()

    # ---------- 4. 零带隙 raw 统计 + L1→L3 端点比较 ----------
    if y is not None:
        zero = (y == 0)
        say("## 4. 零带隙子集 raw 统计（三个 Level）")
        say()
        say("| Level | 来源等级 | n | mean | median | IQR | q5 | q95 | 负值数 | 负值占比 | raw MAE |")
        say("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        tiers = {
            "L1": "frozen-equivalent reconstruction（正值子集直接对齐，负值子集推断）",
            "L2": "old-imputation diagnostic only",
            "L3": "same-run frozen exact",
        }
        for name, arr in [("L1", l1_raw), ("L2", l2_raw), ("L3", l3_raw)]:
            s = dist_stats(arr[zero])
            mae = float(np.abs(arr[zero] - y[zero]).mean())
            say(f"| {name} | {tiers[name]} | {s['n']:,} | {s['mean']:.4f} | {s['median']:.4f} | "
                f"{s['IQR']:.4f} | {s['q5']:.4f} | {s['q95']:.4f} | "
                f"{s['neg_count']:,} | {s['neg_share']:.1%} | {mae:.4f} |")
        say()

        say("## 5. 正式端点比较：Level 1 → Level 3（零带隙子集）")
        say()
        a, b = l1_raw[zero], l3_raw[zero]
        sa, sb = dist_stats(a), dist_stats(b)
        say(f"- IQR：{sa['IQR']:.4f} → {sb['IQR']:.4f} eV，收缩 **{sa['IQR']/sb['IQR']:.1f} 倍**")
        say(f"- median：{sa['median']:.4f} → {sb['median']:.4f} eV "
            f"（分母接近零，倍数对舍入敏感，正文优先引用 IQR）")
        say(f"- raw MAE：{np.abs(a - y[zero]).mean():.4f} → {np.abs(b - y[zero]).mean():.4f} eV")
        say(f"- 负值占比：{sa['neg_share']:.1%} → {sb['neg_share']:.1%}")
        say()

        # 逐样本配对：delta > 0 表示 L3 更接近零。改善不具传递性，必须实算，不预设方向。
        delta = np.abs(a) - np.abs(b)
        imp, wor, tie = int((delta > 0).sum()), int((delta < 0).sum()), int((delta == 0).sum())
        n = delta.size
        say(f"- 逐样本配对（Δ = |L1| − |L3|，Δ>0 表示 L3 更接近零）：")
        say(f"  - 改善 **{imp:,}（{imp/n:.1%}）** / 恶化 {wor:,}（{wor/n:.1%}）/ 持平 {tie:,}")
        say(f"  - Δ 中位数 {np.median(delta):+.4f}，均值 {delta.mean():+.4f}，"
            f"P10 {np.percentile(delta, 10):+.4f}，P90 {np.percentile(delta, 90):+.4f}")
        say()
        say("  逐折改善率：")
        off = 0
        for k in FOLDS:
            sz = fold_sizes[k]
            m = zero[off:off + sz]
            d = np.abs(l1_raw[off:off + sz][m]) - np.abs(l3_raw[off:off + sz][m])
            say(f"  - fold {k}：{ (d>0).mean():.1%}（n={m.sum():,}，Δ中位数 {np.median(d):+.4f}）")
            off += sz
        say()

    # ---------- 6. 结论表 ----------
    say("## 6. Claim status")
    say()
    say("| 主张 | 状态 |")
    say("|---|---|")
    say(f"| L1 正值子集 raw 一致 | {'直接验证通过' if r_pos['exact_unequal'] == 0 else '**未通过**'} |")
    say(f"| L1 全体裁剪后一致 | {'通过' if r_all['exact_unequal'] == 0 else '**未通过**'} |")
    say(f"| L1 负值子集 raw 一致 | 由确定性模型同一性推断"
        f"{'（有逐折日志对照）' if (P_L1_ARCHIVED_LOG and logs_ok) else '（无归档日志，证据弱一档）'} |")
    say(f"| L2 raw 统计 | Diagnostic only |")
    say(f"| L2 冻结 clipping gain | Unavailable |")
    say(f"| L3 raw/clipped 一致 | {'同源精确' if r_l3['exact_unequal'] == 0 else '**未通过**'} |")
    say(f"| L1→L3 端点比较 | {'可用作正式结果' if y is not None else '缺标签，未计算'} |")
    say()

    Path(OUT_MD).write_text("\n".join(report), encoding="utf-8")
    print(f"\n报告已写入 {OUT_MD}")


# ============================================================
# 标签文件生成方法（在你的环境里跑一次，API 以你本地版本为准）
#
#   from matbench.bench import MatbenchBenchmark
#   import numpy as np
#   mb = MatbenchBenchmark(autoload=False, subset=["matbench_mp_gap"])
#   task = list(mb.tasks)[0]; task.load()
#   d = {}
#   for k, fold in enumerate(task.folds):
#       _, y_test = task.get_test_data(fold, include_target=True)
#       d[f"ids_{k}"] = np.asarray(y_test.index, dtype="U16")
#       d[f"y_{k}"]   = np.asarray(y_test.values, dtype=np.float64)
#   np.savez("labels_by_fold.npz", **d)
#
# 跑完后脚本里的 ID 一致性断言会替你确认折序对不对。
# ============================================================

if __name__ == "__main__":
    main()
