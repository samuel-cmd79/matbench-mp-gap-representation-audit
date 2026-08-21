"""
v4 angle-mask fold-0 结果的双版本 matbench 评估 + 官方提交文件生成
  - 版本 A: 原始预测直接记录
  - 版本 B: clip 到 [0, +inf)（物理约束后处理，用于官方提交）

在装有 matbench 的机器上运行（Mac mini 或 Colab 均可）。
需要 result_v4_angle_zero_fold0/fold_0/test_preds.npz（从 Drive 下载或直接挂载）。

    python evaluate_and_clip_v4.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark

# ========= 按实际环境改 =========
RESULTS_DIR = Path('../result_v4_angle_zero_fold0')          # 仅含本对照所需的 fold_0
OUT_RAW     = Path('matbench_v4_angle_zero_raw.json.gz')
OUT_CLIP    = Path('matbench_v4_angle_zero_clipped.json.gz')
FOLDS       = [0]
# ================================

mb_raw  = MatbenchBenchmark(autoload=False, subset=['matbench_mp_gap'])
mb_clip = MatbenchBenchmark(autoload=False, subset=['matbench_mp_gap'])
task_raw  = list(mb_raw.tasks)[0]
task_clip = list(mb_clip.tasks)[0]
task_raw.load()
task_clip.load()

rows = []
for fold in FOLDS:
    npz = np.load(RESULTS_DIR / f'fold_{fold}' / 'test_preds.npz',
                  allow_pickle=True)
    pred_series = pd.Series(npz['preds'].astype(float),
                            index=[str(i) for i in npz['ids']])

    # 用官方 API 的 id 顺序对齐，杜绝任何顺序假设
    test_ids = [str(i) for i in task_raw.get_test_data(fold, as_type='df').index]
    missing = set(test_ids) - set(pred_series.index)
    if missing:
        raise RuntimeError(f'fold {fold}: 缺 {len(missing)} 条预测，例如 '
                           f'{sorted(missing)[:3]}')
    preds = pred_series.reindex(test_ids).values
    assert not np.isnan(preds).any(), f'fold {fold}: 对齐后存在 NaN'

    preds_clip = np.clip(preds, 0.0, None)

    # 记录进两套 benchmark
    task_raw.record(fold, preds)
    task_clip.record(fold, preds_clip)

    # 自算逐 fold 指标（透明、可复核）
    y = task_raw.get_test_data(fold, as_type='df',
                               include_target=True).iloc[:, -1].values
    mae_raw = float(np.mean(np.abs(preds - y)))
    mae_clip = float(np.mean(np.abs(preds_clip - y)))
    n_neg = int((preds < 0).sum())
    rows.append({
        'fold': fold,
        'MAE_raw': mae_raw,
        'MAE_clip': mae_clip,
        'gain': mae_raw - mae_clip,
        'n_negative': n_neg,
        'min_pred': float(preds.min()),
    })
    print(f'fold {fold}: raw {mae_raw:.4f} -> clip {mae_clip:.4f} '
          f'(负预测 {n_neg} 条, 最小值 {preds.min():.3f})')

df = pd.DataFrame(rows)
print('\n' + '=' * 60)
print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('=' * 60)
print(f'RAW : MAE = {df.MAE_raw.mean():.4f} ± {df.MAE_raw.std():.4f}')
print(f'CLIP: MAE = {df.MAE_clip.mean():.4f} ± {df.MAE_clip.std():.4f}')
print(f'clip 平均增益: {df.gain.mean():.4f} eV')
print(f'对照: v1/v2 ≈ 0.33, MAD 基线 = 1.327, ALIGNN 论文 = 0.186')

# matbench 官方口径的汇总（与自算值应一致）
#print('\nmatbench 官方口径:')
#print('  raw :', task_raw.scores['mae'])
#print('  clip:', task_clip.scores['mae'])
# ---- 另存 clip 后的 npz（与原 test_preds.npz 同目录、同格式）----
for fold in FOLDS:
    npz = np.load(RESULTS_DIR / f'fold_{fold}' / 'test_preds.npz',
                  allow_pickle=True)
    np.savez(RESULTS_DIR / f'fold_{fold}' / 'test_preds_clipped.npz',
             ids=npz['ids'],
             preds=np.clip(npz['preds'].astype(float), 0.0, None))
print('已保存各 fold 的 test_preds_clipped.npz')

#mb_raw.to_file(str(OUT_RAW))
#mb_clip.to_file(str(OUT_CLIP))
#print(f'\n已保存: {OUT_RAW}（留档）\n        {OUT_CLIP}（官方提交用这份）')
#print('提交时在 algorithm description 中注明: '
#      '"predictions clipped to [0, inf) as a physical constraint"')
