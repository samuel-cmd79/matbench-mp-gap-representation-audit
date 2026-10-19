"""
生成两个逐条预测文件 (每个协议一个, L1 和 L2 合并成长表):

  grouped_predictions_seed42.csv   分组划分 (读两个分组版 notebook 的 pred_rows_*.csv)
  official_predictions_seed42.csv  官方折   (读 replay 目录的 raw / 截断预测, 不重新训练)

列: protocol, config, seed, fold, id, formula, formula_count_global, y_true, y_raw, y_clip
  - fold 在两个文件里含义不同: 分组文件是 GroupKFold 折号, 官方文件是 matbench 官方折号
  - formula_count_global = 该化学式在全部 106,113 条里出现的次数 (只用于分析, 不是模型特征)

核对:
  - 每个文件里 (config, id) 唯一; 每个 config 正好 106,113 条
  - 同一文件内 L1 与 L2 的 id 集合、y_true、fold 完全一致
  - y_clip == max(y_raw, 0)
  - 官方文件: replay 的截断预测与 run0709 冻结预测逐位一致
  - 两个文件之间同一 id 的 y_true、formula 一致

用法: 两个分组版 notebook 都跑完后, 在 notebooks 目录下运行  python summarize_groupcv.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ==================== 配置区 ====================
SEED       = 42
EXPECTED_N = 106_113
SPLIT_DIR  = Path("../groupcv_splits")
OUT_DIR    = Path("../groupcv_results")

GROUP_PARTS = {
    "L1": Path("../outputs_v1_groupcv") / f"pred_rows_group_L1_seed{SEED}.csv",
    "L2": Path("../matbench_outputs_v2_groupcv") / f"pred_rows_group_L2_seed{SEED}.csv",
}
# 官方折: (replay 目录 — 有 raw, 8000 轮; run0709 冻结目录 — 用来逐位核对)
OFFICIAL = {
    "L1": (Path("../outputs_v1_run0709_replay"),         Path("../outputs_v1_run0709")),
    "L2": (Path("../matbench_outputs_v2_run0709_replay"), Path("../matbench_outputs_v2_run0709")),
}
MODEL = "xgb"
NUM_BOOST_ROUND = 8000      # 主协议上限
COLS = ["protocol", "config", "seed", "fold", "id", "formula", "formula_count_global",
        "y_true", "y_raw", "y_clip"]
# ================================================


def check_table(df, name):
    assert list(df.columns) == COLS, f"{name}: 列不对 {list(df.columns)}"
    assert not df[["y_true", "y_raw", "y_clip"]].isna().any().any(), f"{name}: 有 NaN"
    assert not df.duplicated(["config", "id"]).any(), f"{name}: (config, id) 有重复"
    assert np.array_equal(np.maximum(df["y_raw"].to_numpy(), 0), df["y_clip"].to_numpy()), \
        f"{name}: y_clip ≠ max(y_raw, 0)"
    configs = sorted(df["config"].unique())
    for c in configs:
        assert (df["config"] == c).sum() == EXPECTED_N, f"{name}: {c} 不是 {EXPECTED_N} 条"
    ref = df[df["config"] == configs[0]].set_index("id").sort_index()
    for c in configs[1:]:
        oth = df[df["config"] == c].set_index("id").sort_index()
        assert ref.index.equals(oth.index), f"{name}: {configs[0]} 与 {c} 的 id 集合不同"
        for col in ["y_true", "fold", "formula", "formula_count_global"]:
            assert (ref[col] == oth[col]).all(), f"{name}: {configs[0]} 与 {c} 的 {col} 不一致"
    print(f"  ✅ {name}: {len(df)} 行, 核对通过")


def summary(df, name):
    d = df.assign(ae=(df.y_clip - df.y_true).abs(), se=(df.y_clip - df.y_true) ** 2)
    g = d.groupby(["config", "fold"]).agg(MAE=("ae", "mean"), MSE=("se", "mean"))
    g["RMSE"] = np.sqrt(g.pop("MSE"))
    with pd.option_context("display.float_format", "{:.6f}".format):
        print(f"\n{name} 逐折:")
        print(g)
        print(f"{name} 五折平均:")
        print(g.groupby("config").mean())


def check_official_record(config, replay_dir, fold):
    """读 replay 目录的训练记录, 确认种子和轮数上限 (有记录就查, 不一致就停)"""
    v1 = replay_dir / f"xgb_training_info_fold_{fold + 1}.json"           # v1 格式, 编号从 1 开始
    v2 = [p for p in replay_dir.glob(f"xgb_training_info_*_fold_{fold}.json") if "testmode" not in p.name]
    if v1.exists():
        d = json.loads(v1.read_text(encoding="utf-8"))
        assert d["params"]["seed"] == SEED, f"{config} 官方 fold {fold}: 训练记录 seed={d['params']['seed']}"
        assert d["total_rounds_evaluated"] <= NUM_BOOST_ROUND, \
            f"{config} 官方 fold {fold}: 训练了 {d['total_rounds_evaluated']} 轮, 超过上限 {NUM_BOOST_ROUND}"
        RECORD_NOTES.append(f"{config} 官方 fold {fold}: v1 记录 seed={SEED}, 实际 {d['total_rounds_evaluated']} 轮 ✅")
    elif len(v2) == 1:
        d = json.loads(v2[0].read_text(encoding="utf-8"))
        assert d["seed"] == SEED and d["num_boost_round_max"] == NUM_BOOST_ROUND, \
            f"{config} 官方 fold {fold}: 训练记录 seed={d['seed']}, 上限={d['num_boost_round_max']}"
        RECORD_NOTES.append(f"{config} 官方 fold {fold}: 记录 seed={SEED}, 上限 {NUM_BOOST_ROUND} ✅")
    else:
        RECORD_NOTES.append(f"{config} 官方 fold {fold}: 无训练记录, 来源仅由与 run0709 逐位一致来保证")


RECORD_NOTES = []


def check_group_records(folds):
    """分组版每折的训练记录: 种子、上限、划分指纹都必须和 folds.csv 对上"""
    for config, part in GROUP_PARTS.items():
        for k in range(5):
            p = part.parent / f"xgb_training_info_group_{config}_seed{SEED}_fold_{k}.json"
            d = json.loads(p.read_text(encoding="utf-8"))
            ids = folds.loc[folds["group_fold"] == k, "id"]
            sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
            assert d["seed"] == SEED and d["num_boost_round_max"] == NUM_BOOST_ROUND, f"{p.name}: seed/上限不对"
            assert d["test_ids_sha256"] == sha, f"{p.name}: 测试集与 folds.csv 不对应"
    print("  ✅ 分组版训练记录: seed、上限、测试集划分都与 folds.csv 对应")


def build_official(folds):
    rows = []
    fb = folds.set_index("id")
    for config, (replay_dir, frozen_dir) in OFFICIAL.items():
        for fold in sorted(folds["official_fold"].unique()):
            ids = folds[folds["official_fold"] == fold].sort_values("official_pos")["id"].to_numpy()
            pdir = replay_dir / f"predictions_{MODEL}"
            raw  = np.load(pdir / f"pred_raw_fold_{fold}.npy", allow_pickle=False)
            clip = np.load(pdir / f"pred_fold_{fold}.npy", allow_pickle=False)
            assert len(raw) == len(clip) == len(ids), f"{config} 官方 fold {fold}: 长度不对"
            assert np.array_equal(np.maximum(raw, 0), clip), f"{config} 官方 fold {fold}: clip ≠ max(raw,0)"
            frozen = frozen_dir / f"predictions_{MODEL}" / f"pred_fold_{fold}.npy"
            if not frozen.exists():
                raise FileNotFoundError(f"缺冻结参考 {frozen}; 没有它无法确认官方折预测的来源, 停止")
            assert np.array_equal(np.load(frozen, allow_pickle=False), clip), \
                f"{config} 官方 fold {fold}: replay 与 run0709 冻结预测不一致"
            check_official_record(config, replay_dir, int(fold))
            rows.append(pd.DataFrame({
                "protocol": "official", "config": config, "seed": SEED, "fold": int(fold), "id": ids,
                "formula": fb.loc[ids, "formula"].to_numpy(),
                "formula_count_global": fb.loc[ids, "formula_count_global"].to_numpy(),
                "y_true": fb.loc[ids, "y_true"].to_numpy(np.float64),
                "y_raw": raw.astype(np.float64), "y_clip": clip.astype(np.float64),
            }))
    return pd.concat(rows, ignore_index=True)[COLS]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(SPLIT_DIR / "folds.csv", dtype={"id": str, "formula": str}, float_precision="round_trip", keep_default_na=False)
    assert len(folds) == EXPECTED_N and folds["id"].is_unique

    # ---- 分组 ----
    grouped = pd.concat([pd.read_csv(p, dtype={"id": str, "formula": str}, float_precision="round_trip", keep_default_na=False) for p in GROUP_PARTS.values()],
                        ignore_index=True)[COLS]
    check_table(grouped, "分组划分")
    check_group_records(folds)
    gf = grouped.drop_duplicates("id").set_index("id")
    fb = folds.set_index("id")
    assert (gf.loc[fb.index, "fold"].to_numpy() == fb["group_fold"].to_numpy()).all(), "分组折号与 folds.csv 不一致"

    # ---- 官方 ----
    official = build_official(folds)
    check_table(official, "官方折")
    for line in RECORD_NOTES:
        print("  " + line)

    # ---- 两个文件之间 ----
    of = official.drop_duplicates("id").set_index("id").loc[gf.index]
    assert np.array_equal(of["y_true"].to_numpy(), gf["y_true"].to_numpy()), "两个文件的 y_true 不一致"
    assert (of["formula"] == gf["formula"]).all(), "两个文件的 formula 不一致"
    print("  ✅ 两个文件之间 y_true / formula 一致")

    p1 = OUT_DIR / f"grouped_predictions_seed{SEED}.csv"
    p2 = OUT_DIR / f"official_predictions_seed{SEED}.csv"
    grouped.to_csv(p1, index=False)
    official.to_csv(p2, index=False)
    print(f"\n✅ 已写出 {p1}\n✅ 已写出 {p2}")

    summary(grouped, "分组划分")
    summary(official, "官方折")


if __name__ == "__main__":
    main()
