"""
负预测分析 + 化学家族分层
=========================
A/B 部分吃 RAW 预测:
  - v1/v2 用 E1 确定性重放导出的 raw npy (replay 目录的 pred_raw_fold_*.npy;
    max(raw, 0) 与 run0709 截断预测逐位一致, 运行时会核对)
  - v3 用 ALIGNN 的 raw npz
C 部分吃 run0709 截断预测 (v1/v2) + v3 clip, 按阴离子家族分层 MAE。

v5 改动 (只改数据源和图上文字, 计算逻辑不变):
  1. v1/v2 raw 换成 E1 重放的 raw; 核对 max(raw,0) == run0709 截断预测
  2. 零带隙条数硬核对 = 46,151; 打印中位数/IQR 并与核对值对比; 打印区间外条数
  3. 图例只写 Level 1 / Level 2 / Level 3; 新图另存为 metal_raw_hist_v2.png (不覆盖旧图)
  4. 保存前检查图上所有文字不含禁用词

本版改动:
  1. v3 路径改为实际的 results_v4 npz; 加载走 'preds' key + 'ids' mbid 校验,
     乱序但集合一致 → 自动重排, 集合不一致 → 报错
  2. load 增加逐 fold 长度校验 (所有系列), clip 系列加负值硬校验 (应为 0)
  3. 直方图改用 range= 而非 np.clip, 避免边界人工尖峰; 显示范围和范围外样本数打印供图注使用
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
# RAW: E1 确定性重放导出的 clip 前预测
RAW_V1 = '../outputs_v1_run0709_replay/predictions_{model}/pred_raw_fold_{fold}.npy'
RAW_V2 = '../matbench_outputs_v2_run0709_replay/predictions_{model}/pred_raw_fold_{fold}.npy'
RAW_V3 = '../results_v4/fold_{fold}/test_preds.npz'            # ALIGNN raw
# CLIPPED (run0709, C 部分用; 同时用来核对 RAW_V1/RAW_V2 的来源)
CLIP_V1 = '../outputs_v1_run0709/predictions_{model}/pred_fold_{fold}.npy'
CLIP_V2 = '../matbench_outputs_v2_run0709/predictions_{model}/pred_fold_{fold}.npy'
CLIP_V3 = '../results_v4/fold_{fold}/test_preds_clipped.npz'   # ALIGNN clipped
V3_KEY      = 'preds'   # npz 中预测值的 key
V3_MBID_KEY = 'ids'     # npz 中 mbid 的 key
MODEL  = 'xgb'
FOLDS  = [0, 1, 2, 3, 4]
TASK_NAME = 'matbench_mp_gap'
OUTPUT_DIR = '../results/neg_and_family_v2'
HIST_FILE  = 'metal_raw_hist_v2.png'     # 新文件名, 不覆盖旧图
EXPECTED_ZERO_GAP = 46_151
# 核对值 (中位数, IQR; eV)。L1/L3 来自论文 Table 2; L2 只作参考 (分位数方法未确认)
CHECK_VALUES = {'v1_raw': (0.0167, 0.1356), 'v2_raw': (0.0104, 0.0975), 'v3_raw': (0.0004, 0.0038)}
EXPECTED_OUTSIDE = {'v1_raw': 1712, 'v2_raw': 1485, 'v3_raw': 1058}
FORBIDDEN_WORDS = ['frozen', 'canonical', 'historical', 'diagnostic', 'old imputation',
                   'old-imputation', 'oracle bound', 'robust undercut']
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

# raw 来源核对: E1 重放的 max(raw, 0) 必须与 run0709 截断预测逐位一致
for rk, ck in [('v1_raw', 'v1'), ('v2_raw', 'v2')]:
    if rk in raw and ck in clip:
        if not np.array_equal(np.maximum(raw[rk], 0), clip[ck]):
            raise ValueError(f'{rk}: max(raw, 0) 与 run0709 截断预测不一致, 请检查 raw 来源')
        print(f'  ✅ {rk}: max(raw, 0) 与 run0709 截断预测逐位一致')
    else:
        raise ValueError(f'缺 {rk} 或 {ck}, 无法核对 raw 来源')

# clip 系列硬校验: 截断后不应有负预测
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
if int(m.sum()) != EXPECTED_ZERO_GAP:
    raise ValueError(f'零带隙条数 {int(m.sum())} ≠ {EXPECTED_ZERO_GAP}, 先检查对齐和筛选')
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

# 核对值对比 (同一算法: np.median / np.percentile 默认线性插值, 五折合并)
print(f'\n核对值 (中位数 / IQR, eV; 保留 4 位比较):')
for ver, (ref_med, ref_iqr) in CHECK_VALUES.items():
    if ver not in raw:
        continue
    pm = raw[ver][m]
    med = float(np.median(pm)); iqr = float(np.percentile(pm, 75) - np.percentile(pm, 25))
    ok = round(med, 4) == ref_med and round(iqr, 4) == ref_iqr
    tag = '✅' if ok else ('⚠️ 参考值, 以脚本输出为准' if ver == 'v2_raw' else '❌ 先检查数据对齐和零带隙筛选')
    print(f'  {ver}: 脚本 {med:.4f} / {iqr:.4f}  vs  核对 {ref_med:.4f} / {ref_iqr:.4f}  {tag}')

# 配对前提: v1/v2 来自 E1 重放, 与官方 get_test_data 同序 (已用 run0709 截断预测核对);
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
LABELS = {'v1_raw': 'Level 1', 'v2_raw': 'Level 2', 'v3_raw': 'Level 3'}
SHORT  = {'v1_raw': 'L1', 'v2_raw': 'L2', 'v3_raw': 'L3'}
n_outside = {}
for ver, p in raw.items():
    pm = p[m]
    n_outside[ver] = int(((pm < HIST_LO) | (pm > HIST_HI)).sum())
    ax.hist(pm, bins=90, range=(HIST_LO, HIST_HI), alpha=0.5,
            label=LABELS.get(ver, ver), color=colors.get(ver), density=True)
ax.axvline(0, color='black', lw=0.8, ls='--')
outside_txt = ', '.join(f'{SHORT.get(v, v)}: {n}' for v, n in n_outside.items())
ax.set_xlabel('Raw prediction (eV)')
ax.set_ylabel('Density')
ax.set_title('Raw predictions on zero-gap materials')
ax.legend()
fig.tight_layout()
print(f'\n直方图显示区间 [{HIST_LO}, {HIST_HI}] eV 之外的条数: ' +
      ', '.join(f'{SHORT.get(v, v)} = {n}' for v, n in n_outside.items()))
for ver, reference in EXPECTED_OUTSIDE.items():
    actual = n_outside.get(ver)
    status = 'MATCH' if actual == reference else 'MISMATCH'
    print(f'  {SHORT[ver]} 区间外条数: actual={actual}; reference={reference}; {status}')
print(f'Fig. S6 caption note: Display range [{HIST_LO}, {HIST_HI}] eV; '
      f'predictions outside the display range: {outside_txt}.')
fig_text = ' '.join(t.get_text() for t in fig.findobj(match=lambda a: hasattr(a, 'get_text'))).lower()
bad = [w for w in FORBIDDEN_WORDS if w in fig_text]
if bad:
    raise ValueError(f'图上出现禁用词: {bad}')
out_path = os.path.join(OUTPUT_DIR, HIST_FILE)
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"→ {out_path}")

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
