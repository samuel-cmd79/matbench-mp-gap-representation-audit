"""
R² 计算 — v1/v2 (rf, xgb) + v3 (GNN), 官方五折 test
====================================================
两种口径都报:
  1. 逐折 R² + 五折 mean ± std   (MatBench 官方报告口径)
  2. 五折拼接后的 pooled R²      (与本项目误差分析各表同口径)
另附 non-metal (Eg>0) 与 metal (Eg=0) 子集的 pooled R² 供参考。
注意: metal 子集内 y 全为 0, 方差为 0, R² 无定义 → 该行只报 MAE。

npz (GNN) 加载: 'preds' key + 'ids' mbid 校验, 乱序自动重排。
"""

import numpy as np
import pandas as pd
import os

# ================= CONFIG =================
V1_PRED_PATTERN = '../outputs_v1_run0709/predictions_{model}/pred_fold_{fold}.npy'
V2_PRED_PATTERN = '../matbench_outputs_v2_run0709/predictions_{model}/pred_fold_{fold}.npy'
V3_PRED_PATTERN = '../results_v4/fold_{fold}/test_preds_clipped.npz'
V3_KEY      = 'preds'
V3_MBID_KEY = 'ids'
FOLDS  = [0, 1, 2, 3, 4]
TASK_NAME = 'matbench_mp_gap'
# ==========================================

SERIES = [
    ('v1-rf',  'rf',  V1_PRED_PATTERN),
    ('v1-xgb', 'xgb', V1_PRED_PATTERN),
    ('v2-rf',  'rf',  V2_PRED_PATTERN),
    ('v2-xgb', 'xgb', V2_PRED_PATTERN),
    ('v3-gnn', None,  V3_PRED_PATTERN),
]

print('加载 MatBench 官方标签...')
from matbench.bench import MatbenchBenchmark
mb = MatbenchBenchmark(autoload=False)
task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
task.load()
target_col = task.metadata['target']

y_by_fold, mbid_by_fold = {}, {}
for fold in FOLDS:
    df = task.get_test_data(fold, as_type='df', include_target=True)
    y_by_fold[fold] = df[target_col].to_numpy(dtype=float)
    mbid_by_fold[fold] = np.array([str(i) for i in df.index], dtype=object)


def load_fold(pattern, model, fold):
    p = pattern.format(model=model, fold=fold) if model else pattern.format(fold=fold)
    if not os.path.exists(p):
        raise FileNotFoundError(f'缺预测文件: {p}')
    if p.endswith('.npz'):
        d = np.load(p, allow_pickle=True)
        arr = np.asarray(d[V3_KEY]).ravel().astype(np.float64)
        ids = np.array([str(i) for i in d[V3_MBID_KEY].ravel()], dtype=object)
        ref = mbid_by_fold[fold]
        if len(arr) != len(ref):
            raise ValueError(f'{p}: 长度 {len(arr)} != 官方 test 数 {len(ref)}')
        if not np.array_equal(ids, ref):
            if set(ids) == set(ref):
                order = pd.Series(np.arange(len(ids)), index=ids)
                arr = arr[order.loc[ref].to_numpy()]
                print(f'  ⚠️ fold{fold}: mbid 顺序不一致, 已按官方顺序重排')
            else:
                raise ValueError(f'{p}: mbid 集合与官方 test 不一致!')
        return arr
    arr = np.load(p).ravel().astype(np.float64)
    if len(arr) != len(y_by_fold[fold]):
        raise ValueError(f'{p}: 长度 {len(arr)} != 官方 test 数 {len(y_by_fold[fold])}')
    return arr


def r2(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ---------- 逐折 R² ----------
print('\n' + '=' * 78)
print('1. 逐折 R² (MatBench 官方报告口径)')
print(f'{"系列":<9}' + ''.join(f' {"fold"+str(f):>9}' for f in FOLDS) +
      f' {"mean":>9} {"std":>8}')
print('-' * 78)
preds_by_fold = {}
for name, model, pattern in SERIES:
    fold_r2 = []
    parts = {}
    for fold in FOLDS:
        arr = load_fold(pattern, model, fold)
        parts[fold] = arr
        fold_r2.append(r2(y_by_fold[fold], arr))
    preds_by_fold[name] = parts
    print(f'{name:<9}' + ''.join(f' {v:>9.4f}' for v in fold_r2) +
          f' {np.mean(fold_r2):>9.4f} {np.std(fold_r2):>8.4f}')

# ---------- pooled R² ----------
y_all = np.concatenate([y_by_fold[f] for f in FOLDS])
nm = y_all > 0
print('\n' + '=' * 78)
print('2. 五折拼接 pooled R² (与误差分析各表同口径)')
print(f'{"系列":<9} {"R²_all":>9} {"R²_nonmetal":>12} {"MAE_metal":>10}')
print('   (metal 子集 y 全为 0, 方差为 0 → R² 无定义, 报 MAE 代替)')
print('-' * 78)
for name, _, _ in SERIES:
    p_all = np.concatenate([preds_by_fold[name][f] for f in FOLDS])
    r2_all = r2(y_all, p_all)
    r2_nm  = r2(y_all[nm], p_all[nm])
    mae_metal = np.abs(p_all[~nm]).mean()
    print(f'{name:<9} {r2_all:>9.4f} {r2_nm:>12.4f} {mae_metal:>10.4f}')

print('\n判读: 逐折 mean±std 用于与 MatBench 榜面对齐; pooled 用于和本项目')
print('bin/家族/polymorph 各表引用同一口径。全集 R² 会被金属大簇 (y=0) 抬高,')
print('non-metal R² 更能反映带隙数值回归本身的解释力, 报正文时建议两个都给。')
