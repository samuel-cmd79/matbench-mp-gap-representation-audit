"""
负预测分析 (pre-freeze 诊断 run) + 化学家族分层 (冻结版)
=========================================================
A/B 部分吃 RAW 预测:
  - v1/v2 用旧 run 的 raw npy (pre-freeze diagnostic; 旧 run 与 run0709 同一管线,
    仅改填充策略与 clip, 均按 matbench 官方 5-fold get_test_data 顺序 → 配对可靠)
  - v3 用 ALIGNN 的 raw npz (与冻结模型同源, 证据强度更高)
C 部分吃冻结版 clipped 预测 (v1/v2 run0709 + v3 clip), 按阴离子家族分层 MAE。

本版改动:
  1. v3 路径改为实际的 results_v4 npz; 加载走 'preds' key + 'ids' mbid 校验,
     乱序但集合一致 → 自动重排, 集合不一致 → 报错
  2. load 增加逐 fold 长度校验 (所有系列), clip 系列加负值硬校验 (应为 0)
  3. 直方图改用 range= 而非 np.clip, 避免边界人工尖峰; 轴标签注明范围外样本数
  4. C 表每家族增加 non-metal-only MAE 列, 剥离金属占比混杂
  5. family 优先级注释明确 (halide > oxide > chalcogenide > pnictide);
     兜底类改名 'other' (含金属间化合物、氢化物、碳化物、硼化物等)

输出: 终端三张表 + 金属子集 raw 预测叠加直方图。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= CONFIG =================
# RAW (诊断用)
RAW_V1 = '../matbench_outputs/v1_predictions_{model}/pred_fold_{fold}.npy'   # 旧 run
RAW_V2 = '../matbench_outputs/v2_predictions_{model}/pred_fold_{fold}.npy'  # 旧 run
RAW_V3 = '../results_v4/fold_{fold}/test_preds.npz'            # ALIGNN raw
# CLIPPED (冻结版, C 部分用)
CLIP_V1 = '../outputs_v1_run0709/predictions_{model}/pred_fold_{fold}.npy'
CLIP_V2 = '../matbench_outputs_v2_run0709/predictions_{model}/pred_fold_{fold}.npy'
CLIP_V3 = '../results_v4/fold_{fold}/test_preds_clipped.npz'   # ALIGNN clipped
V3_KEY      = 'preds'   # npz 中预测值的 key
V3_MBID_KEY = 'ids'     # npz 中 mbid 的 key
MODEL  = 'xgb'
FOLDS  = [0, 1, 2, 3, 4]
TASK_NAME = 'matbench_mp_gap'
OUTPUT_DIR = './neg_and_family'
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('加载 MatBench 标签与结构元素...')
from matbench.bench import MatbenchBenchmark
mb = MatbenchBenchmark(autoload=False)
task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
task.load()
target_col = task.metadata['target']

ys, mbid_parts, elems = [], [], []
for fold in FOLDS:
    df = task.get_test_data(fold, as_type='df', include_target=True)
    ys.append(df[target_col].to_numpy(dtype=float))
    mbid_parts.append(np.array([str(i) for i in df.index], dtype=object))
    elems += [set(str(el) for el in r.composition.elements) for r in df['structure']]
y = np.concatenate(ys)


def load(pattern, model=None, name=''):
    """拼接五折预测; npz 走 key + mbid 校验/重排, npy 做长度校验。缺文件返回 None。"""
    parts = []
    for fold in FOLDS:
        p = pattern.format(model=model, fold=fold) if model else pattern.format(fold=fold)
        if not os.path.exists(p):
            print(f'  ⚠️ 跳过 {name}: 缺文件 {p}')
            return None
        if p.endswith('.npz'):
            d = np.load(p, allow_pickle=True)
            if V3_KEY not in d.files or V3_MBID_KEY not in d.files:
                raise KeyError(f'{p}: 需要 keys "{V3_KEY}"/"{V3_MBID_KEY}", 实际 = {d.files}')
            arr = np.asarray(d[V3_KEY]).ravel().astype(np.float64)
            ids = np.array([str(i) for i in d[V3_MBID_KEY].ravel()], dtype=object)
            ref = mbid_parts[fold]
            if len(arr) != len(ref):
                raise ValueError(f'{p}: 长度 {len(arr)} != 官方 test 数 {len(ref)}')
            if not np.array_equal(ids, ref):
                if set(ids) == set(ref):
                    order = pd.Series(np.arange(len(ids)), index=ids)
                    arr = arr[order.loc[ref].to_numpy()]
                    print(f'  ⚠️ {name} fold{fold}: mbid 顺序不一致, 已按官方顺序重排')
                else:
                    raise ValueError(f'{p}: mbid 集合与官方 test 不一致!')
        else:
            arr = np.load(p).ravel().astype(np.float64)
            if len(arr) != len(ys[fold]):
                raise ValueError(f'{p}: 长度 {len(arr)} != 官方 test 数 {len(ys[fold])}')
        parts.append(arr)
    return np.concatenate(parts)


raw = {k: v for k, v in {
    'v1_raw': load(RAW_V1, MODEL, 'v1_raw'), 'v2_raw': load(RAW_V2, MODEL, 'v2_raw'),
    'v3_raw': load(RAW_V3, name='v3_raw')}.items() if v is not None}
clip = {k: v for k, v in {
    'v1': load(CLIP_V1, MODEL, 'v1'), 'v2': load(CLIP_V2, MODEL, 'v2'),
    'v3': load(CLIP_V3, name='v3')}.items() if v is not None}
print(f'raw 系列: {list(raw.keys())} | clip 系列: {list(clip.keys())}')

# clip 系列硬校验: 冻结版不应有负预测
for ver, p in clip.items():
    n_neg = int((p < 0).sum())
    if n_neg:
        print(f'  ⚠️ clip 系列 {ver} 有 {n_neg} 个负预测——确认文件是否为冻结/clipped 版本!')

# ---------- A. 负预测按真实带隙分箱 ----------
NEG_BINS = [('Eg = 0', y == 0), ('0 < Eg ≤ 0.1', (y > 0) & (y <= 0.1)),
            ('0.1 < Eg ≤ 0.5', (y > 0.1) & (y <= 0.5)), ('Eg > 0.5', y > 0.5)]
print('\n' + '=' * 80)
print('A. 负预测的真实带隙分布 (raw 预测)')
print(f'{"版本":<8} {"负预测总数":>10} {"占比":>7} | ' +
      ' | '.join(f'{lab}' for lab, _ in NEG_BINS))
print('-' * 80)
for ver, p in raw.items():
    neg = p < 0
    parts = []
    for lab, mask in NEG_BINS:
        parts.append(f'{int((neg & mask).sum()):>6} ({(neg & mask).sum()/max(neg.sum(),1):.0%})')
    print(f'{ver:<8} {int(neg.sum()):>10} {neg.mean():>6.1%} | ' + ' | '.join(parts))
print('\n各真值区间内被预测为负的比例:')
for ver, p in raw.items():
    parts = [f'{lab}: {(p[mask] < 0).mean():.1%}' for lab, mask in NEG_BINS]
    print(f'  {ver}: ' + ' | '.join(parts))

# ---------- B. 金属子集 raw 统计 + 配对 Δ ----------
m = y == 0
print('\n' + '=' * 80)
print(f'B. 金属子集 (Eg=0, n={int(m.sum())}) raw 预测统计')
rows = {}
for ver, p in raw.items():
    pm = p[m]
    rows[ver] = {
        'mean': pm.mean(), 'median': np.median(pm), 'std': pm.std(),
        'IQR': np.percentile(pm, 75) - np.percentile(pm, 25),
        'q5': np.percentile(pm, 5), 'q25': np.percentile(pm, 25),
        'q75': np.percentile(pm, 75), 'q95': np.percentile(pm, 95),
        'neg_frac': (pm < 0).mean(), 'raw_MAE(=mean|pred|)': np.abs(pm).mean(),
    }
print(pd.DataFrame(rows).to_string(float_format=lambda x: f'{x:+.4f}'))

# 配对前提: v1/v2 旧 run 与官方 get_test_data 同序 (同一管线, 仅改填充与 clip);
# v3 npz 已经过 mbid 校验/重排 → 三方均按官方顺序, 逐样本配对成立
pairs = [(a, b) for a, b in [('v1_raw', 'v2_raw'), ('v2_raw', 'v3_raw')]
         if a in raw and b in raw]
print('\n配对分析 Δi = |pred_A| − |pred_B| (金属真值=0, Δ>0 表示 B 更近零):')
for a, b in pairs:
    d = np.abs(raw[a][m]) - np.abs(raw[b][m])
    print(f'  {a} vs {b}: Δ>0 占 {(d > 0).mean():.1%}, Δ 中位数 {np.median(d):+.4f} eV')

# 直方图 (range= 截取显示范围, 范围外样本不画, 不造边界尖峰)
HIST_LO, HIST_HI = -0.6, 1.2
fig, ax = plt.subplots(figsize=(8, 4))
colors = {'v1_raw': '#85B7EB', 'v2_raw': '#185FA5', 'v3_raw': '#C0392B'}
LABELS = {'v1_raw': f'Level 1 ({MODEL.upper()}, raw)',
          'v2_raw': f'Level 2 ({MODEL.upper()}, raw) — old-imputation diagnostic',
          'v3_raw': 'Level 3 (ALIGNN, raw)'}
SHORT  = {'v1_raw': 'L1', 'v2_raw': 'L2', 'v3_raw': 'L3'}
n_outside = {}
for ver, p in raw.items():
    pm = p[m]
    n_outside[ver] = int(((pm < HIST_LO) | (pm > HIST_HI)).sum())
    ax.hist(pm, bins=90, range=(HIST_LO, HIST_HI), alpha=0.5,
            label=LABELS.get(ver, ver), color=colors.get(ver), density=True)
ax.axvline(0, color='black', lw=0.8, ls='--')
outside_txt = ', '.join(f'{SHORT.get(v, v)}: {n}' for v, n in n_outside.items())
ax.set_xlabel(f'Raw prediction for true zero-gap entries (eV; display range [{HIST_LO}, {HIST_HI}], '
              f'outside: {outside_txt})')
ax.set_ylabel('Density')
ax.set_title('Raw predictions on zero-gap materials')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'metal_raw_hist.png'), dpi=200)
plt.close(fig)
print(f"→ {os.path.join(OUTPUT_DIR, 'metal_raw_hist.png')}")

# ---------- C. 化学家族分层 (冻结版 clipped) ----------
# 优先级: halide > oxide > chalcogenide > pnictide > other
# (氟氧化物 → halide, 氮氧化物 → oxide; 'other' 含金属间化合物/氢化物/碳化物/硼化物等)
HALOGENS, CHALC, PNIC = {'F','Cl','Br','I'}, {'S','Se','Te'}, {'N','P','As','Sb','Bi'}
def family(els):
    if els & HALOGENS: return 'halide'
    if 'O' in els:     return 'oxide'
    if els & CHALC:    return 'chalcogenide'
    if els & PNIC:     return 'pnictide'
    return 'other'
fam = np.array([family(e) for e in elems])

print('\n' + '=' * 80)
print('C. 化学家族分层 MAE (冻结版 clipped 预测; nm = non-metal-only, 剥离金属占比混杂)')
hdr = ' | '.join(f'MAE_{v} MAE_{v}_nm' for v in clip)
print(f'{"家族":<14} {"n":>7} {"金属占比":>8} | {hdr}')
print('-' * 80)
for f_ in ['oxide', 'halide', 'chalcogenide', 'pnictide', 'other']:
    mask = fam == f_
    mask_nm = mask & (y > 0)
    cells = []
    for v in clip:
        mae_all = np.abs(y[mask] - clip[v][mask]).mean()
        mae_nm  = (np.abs(y[mask_nm] - clip[v][mask_nm]).mean()
                   if mask_nm.any() else np.nan)
        cells.append(f'{mae_all:>6.4f} {mae_nm:>9.4f}')
    print(f'{f_:<14} {int(mask.sum()):>7} {(y[mask]==0).mean():>7.1%} | '
          + ' | '.join(cells))
print('\n完成。')
