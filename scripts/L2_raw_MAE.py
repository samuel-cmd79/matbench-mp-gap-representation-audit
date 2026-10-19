import numpy as np
from pathlib import Path
from matbench.bench import MatbenchBenchmark

REPLAY = Path("../matbench_outputs_v2_run0709_replay")   # L2 重放 (8000 轮, 有 raw)
FROZEN = Path("../matbench_outputs_v2_run0709")          # 冻结版本, 用来核对来源
CLIP_MAE_REF   = 0.1462       # 你给的截断后 MAE
EXPECTED_ZEROS = 46_151

mb = MatbenchBenchmark(autoload=False)
task = next(t for t in mb.tasks if t.dataset_name == "matbench_mp_gap")
task.load()
tcol = task.metadata["target"].strip()

raws, clips, ys = [], [], []
for fold in task.folds:
    df = task.get_test_data(fold, as_type="df", include_target=True).rename(columns=lambda x: x.strip())
    y = df[tcol].to_numpy(np.float64)
    raw  = np.load(REPLAY / "predictions_xgb" / f"pred_raw_fold_{fold}.npy", allow_pickle=False).astype(np.float64)
    clip = np.load(REPLAY / "predictions_xgb" / f"pred_fold_{fold}.npy",     allow_pickle=False)
    frz  = np.load(FROZEN / "predictions_xgb" / f"pred_fold_{fold}.npy",     allow_pickle=False)
    assert len(raw) == len(clip) == len(y), f"fold {fold}: 长度不对"
    assert np.array_equal(np.maximum(raw, 0), clip.astype(np.float64)), f"fold {fold}: clip ≠ max(raw,0)"
    assert np.array_equal(clip, frz), f"fold {fold}: 重放与 run0709 冻结预测不一致"
    raws.append(raw); clips.append(clip.astype(np.float64)); ys.append(y)

raw, clip, y = map(np.concatenate, (raws, clips, ys))
zero = y == 0
n0 = int(zero.sum())
print(f"零带隙条数: {n0}  (预期 {EXPECTED_ZEROS}) {'✅' if n0 == EXPECTED_ZEROS else '❌ 先查口径'}")

r0 = raw[zero]
q1, med, q3 = np.percentile(r0, [25, 50, 75])
raw_mae  = float(np.mean(np.abs(r0)))            # 真值都是 0, 所以就是 |raw| 的均值
clip_mae = float(np.mean(np.abs(clip[zero])))    # 同一批样本, 截断后

print(f"\nraw 中位数 : {med:.6f} eV")
print(f"raw IQR    : {q3 - q1:.6f} eV   (Q1 = {q1:.6f}, Q3 = {q3:.6f})")
print(f"raw MAE    : {raw_mae:.6f} eV")
print(f"截断后 MAE (脚本自算): {clip_mae:.6f} eV   vs 你给的 {CLIP_MAE_REF}  "
      f"{'✅ 一致' if round(clip_mae, 4) == CLIP_MAE_REF else '❌ 不一致, 0.1462 可能不是同一口径'}")
print(f"\n截断收益 = raw MAE − {CLIP_MAE_REF} = {raw_mae - CLIP_MAE_REF:.6f} eV")
print(f"截断收益 (用自算的截断 MAE) = {raw_mae - clip_mae:.6f} eV")
print(f"  恒等式核对: 负值部分的平均幅度 = {np.mean(np.maximum(-r0, 0)):.6f} eV (应与上一行相同)")