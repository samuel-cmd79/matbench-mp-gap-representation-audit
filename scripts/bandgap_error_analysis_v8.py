"""
Band Gap Error Analysis v6 — 官方 fold test 版 + ALIGNN
====================================================
主要功能:
  1. 新增 v3 = ALIGNN 预测 (results_v4/fold_{fold}/test_preds_clipped.npz, 已 clip)
  2. npz 加载: key 'preds' 为预测, 'ids' 为 mbid, 逐 fold 与官方 test 索引校验;
     乱序但集合一致 → 自动按官方顺序重排 (打印警告), 集合不一致 → 报错
  3. delta 双份: v1→v2 (结构特征增益), v2→v3 (ALIGNN 相对树模型增益)
  4. CSV / 终端摘要 / 图1 / 图2 均加入 v3; 图4 改为 v1→v2 vs v2→v3 (XGB 基线)
  5. 三版预测均已 clip at zero → 负预测统计一律为硬校验 (应为 0)
  6. top50 误差样本导出扩展到 v3
  7. 新增 L1-XGB / L2-XGB / L3-ALIGNN 的三子集绝对误差分布表:
     ALL / zero-gap / non-zero-gap 各报精确零误差率、P50/P90/P95/P99/P99.5/max
  8. 新增三面板绝对误差分位数曲线; symlog 保留 clipping 产生的零误差原子,
     并直接标出 P50 与 P99
  9. 新增 top 1% SSE 的绝对量与占比, 区分“尾部绝对恶化”和“尾部相对集中”
 10. MAE 分箱图另输出三层代表系列版: L1-XGB / L2-XGB / L3-ALIGNN

用法: 改 CONFIG, python bandgap_error_analysis_v6.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — 只需修改这里 ({model} → rf/xgb, {fold} → 0-4)
# ============================================================
V1_PRED_PATTERN = '../outputs_v1_run0709/predictions_{model}/pred_fold_{fold}.npy'
V2_PRED_PATTERN = '../matbench_outputs_v2_run0709/predictions_{model}/pred_fold_{fold}.npy'
V3_PRED_PATTERN = '../results_v4/fold_{fold}/test_preds_clipped.npz'   # ALIGNN, 已 clip
V3_KEY      = 'preds'   # npz 中预测值的 key
V3_MBID_KEY = 'ids'     # npz 中 mbid 的 key (None = 不做顺序校验)
OUTPUT_DIR = './error_analysis_official'
TASK_NAME  = 'matbench_mp_gap'

# 分位数图的 symlog 线性区间。0--1 meV 保持线性, 从而能显示精确零误差。
QUANTILE_SYMLOG_LINTHRESH = 1e-3
TAIL_FRACTION = 0.01
# ============================================================

MODELS = ['rf', 'xgb']          # 树模型 (v1/v2 各有两个)
FOLDS  = [0, 1, 2, 3, 4]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (version, model, pattern) — v3 只有 ALIGNN 一个模型
SERIES = [
    ('v1', 'rf',  V1_PRED_PATTERN),
    ('v1', 'xgb', V1_PRED_PATTERN),
    ('v2', 'rf',  V2_PRED_PATTERN),
    ('v2', 'xgb', V2_PRED_PATTERN),
    ('v3', 'gnn', V3_PRED_PATTERN),
]

# 三层分布比较采用与图4相同的 XGB 基线。Level 1/2 不能只写 Level 而省略模型名。
DISTRIBUTION_SERIES = [
    ('Level 1 (XGB)', ('v1', 'xgb')),
    ('Level 2 (XGB)', ('v2', 'xgb')),
    ('Level 3 (ALIGNN)', ('v3', 'gnn')),
]

# ---------- bin 定义: 金属单独, 其余左开右闭 ----------
METAL_LABEL = 'zero gap (0)'
BIN_EDGES   = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, np.inf]   # 用于 y>0, right=True
BIN_LABELS  = ['(0,0.5]', '(0.5,1]', '(1,2]', '(2,3]', '(3,5]', '(5,8]', '8+']
ALL_BINS    = [METAL_LABEL] + BIN_LABELS
SMALL_N_WARN = 100   # n 低于此值的 bin 在轴标签直接标 n


def assign_bins(y_true: np.ndarray) -> pd.Categorical:
    """Eg == 0 → zero gap; Eg > 0 → (0,0.5], (0.5,1], ... 8+"""
    labels = pd.Series(pd.cut(y_true, bins=BIN_EDGES, labels=BIN_LABELS,
                              right=True, include_lowest=False).astype(object))
    labels[y_true == 0] = METAL_LABEL
    return pd.Categorical(labels, categories=ALL_BINS, ordered=True)


# ============================================================
# 加载: 官方标签 + 五折预测拼接
# ============================================================
print('=' * 60)
print('加载 MatBench 官方标签与五折预测...')
print('=' * 60)

from matbench.bench import MatbenchBenchmark
mb = MatbenchBenchmark(autoload=False)
task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
task.load()
target_col = task.metadata['target']

y_parts, mbid_parts = [], []
for fold in FOLDS:
    df = task.get_test_data(fold, as_type='df', include_target=True)
    y_parts.append(df[target_col].to_numpy(dtype=float))
    mbid_parts.append(np.array([str(i) for i in df.index], dtype=object))
y_true = np.concatenate(y_parts)
mbids  = np.concatenate(mbid_parts)
print(f'官方五折 test 拼接: {len(y_true)} 样本, 金属 (Eg=0) 占 {(y_true == 0).mean():.1%}')


def load_fold_pred(pattern, model, fold):
    """统一加载一个 fold 的预测, 返回 (pred_1d_float64, mbid_or_None)"""
    p = pattern.format(model=model, fold=fold)
    if not os.path.exists(p):
        raise FileNotFoundError(f'缺预测文件: {p}')
    if p.endswith('.npz'):
        d = np.load(p, allow_pickle=True)
        if V3_KEY not in d.files:
            raise KeyError(f'{p}: 没有 key "{V3_KEY}", 实际 keys = {d.files}')
        pred = np.asarray(d[V3_KEY]).ravel().astype(np.float64)
        mbid = None
        if V3_MBID_KEY is not None:
            if V3_MBID_KEY not in d.files:
                raise KeyError(f'{p}: 没有 key "{V3_MBID_KEY}", 实际 keys = {d.files}')
            mbid = np.array([str(i) for i in d[V3_MBID_KEY].ravel()], dtype=object)
        return pred, mbid
    return np.load(p).ravel().astype(np.float64), None


preds = {}   # (version, model) -> concatenated predictions (官方顺序)
for ver, model, pattern in SERIES:
    parts = []
    for fold in FOLDS:
        arr, fold_mbid = load_fold_pred(pattern, model, fold)
        if len(arr) != len(y_parts[fold]):
            raise ValueError(f'{ver}-{model} fold{fold}: '
                             f'长度 {len(arr)} != 该 fold 标签数 {len(y_parts[fold])}')
        # 有 mbid 就做顺序校验 (ALIGNN dataloader 乱序的关键防线)
        if fold_mbid is not None:
            if not np.array_equal(fold_mbid, mbid_parts[fold]):
                if set(fold_mbid) == set(mbid_parts[fold]):
                    order = pd.Series(np.arange(len(fold_mbid)), index=fold_mbid)
                    arr = arr[order.loc[mbid_parts[fold]].to_numpy()]
                    print(f'  ⚠️ {ver}-{model} fold{fold}: mbid 顺序不一致, 已按官方顺序重排')
                else:
                    raise ValueError(f'{ver}-{model} fold{fold}: '
                                     f'mbid 集合与官方 test 不一致! 确认 fold 对应关系')
        parts.append(arr)
    pred = np.concatenate(parts)
    preds[(ver, model)] = pred
    n_neg = int((pred < 0).sum())
    if n_neg:
        raise ValueError(f'{ver}-{model}: 仍有 {n_neg} 个负预测; '
                         '本脚本的分布比较要求三层全部使用 clipped 预测')
    print(f'  {ver}-{model}: {len(pred)} 条, 负预测校验: OK (已clip)')


# 三个子集使用完全相同的样本定义。P(error=0) 使用严格等于零,
# 不用 isclose, 因为这里要统计的正是 clipping 产生的点质量（atom）。
DISTRIBUTION_SUBSETS = [
    ('ALL', np.ones(len(y_true), dtype=bool)),
    ('zero-gap (Eg=0)', y_true == 0),
    ('non-zero-gap (Eg>0)', y_true > 0),
]


# ============================================================
# 指标计算
# ============================================================

def compute_bin_metrics(y_true, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['abs_err'] = np.abs(df['y_true'] - df['y_pred'])
    df['err']     = df['y_pred'] - df['y_true']
    df['bin']     = assign_bins(y_true)

    def row(sub, label):
        ae, err = sub['abs_err'], sub['err']
        return {
            'bin': label, 'n': len(sub),
            'MAE': ae.mean(), 'RMSE': np.sqrt((err ** 2).mean()),
            'MedianAE': ae.median(), 'P90AE': ae.quantile(0.90), 'P95AE': ae.quantile(0.95),
            'Bias': err.mean(),
        }

    rows = [row(df[df['bin'] == b], b) for b in ALL_BINS if (df['bin'] == b).any()]
    rows.append(row(df, 'ALL'))
    rows.append(row(df[df['y_true'] > 0], 'ALL non-metal'))
    rows.append(row(df[df['y_true'] == 0], 'ALL metal'))
    return pd.DataFrame(rows).set_index('bin')


def compute_delta(m1, m2):
    delta = pd.DataFrame(index=m1.index)
    for c in ['MAE', 'RMSE', 'MedianAE', 'P90AE', 'P95AE', 'Bias']:
        delta[f'Δ{c}'] = m2[c] - m1[c]
    delta['MAE_improvement_%']  = (m1['MAE'] - m2['MAE']) / m1['MAE'] * 100
    delta['RMSE_improvement_%'] = (m1['RMSE'] - m2['RMSE']) / m1['RMSE'] * 100
    return delta


def compute_distribution_row(y_sub, pred_sub, subset, level):
    """汇总一个 level × subset 的绝对误差分布与 top-1% SSE。"""
    if len(y_sub) == 0:
        raise ValueError(f'{level} / {subset}: 子集为空')

    abs_err = np.abs(y_sub - pred_sub)
    sq_err = abs_err ** 2
    exact_zero = (abs_err == 0.0)  # 刻意使用严格相等: 统计 clipping 形成的原子
    boundary_zero = (y_sub == 0.0) & (pred_sub == 0.0)

    # 固定取误差平方最大的 ceil(1% * n) 个样本。既报绝对 SSE, 也报其相对占比;
    # 后者会受总 SSE 下降影响, 不能单独当作“尾部绝对恶化”的证据。
    top_n = max(1, int(np.ceil(TAIL_FRACTION * len(sq_err))))
    if top_n == len(sq_err):
        top_sq_err = sq_err
    else:
        top_idx = np.argpartition(sq_err, -top_n)[-top_n:]
        top_sq_err = sq_err[top_idx]
    total_sse = float(sq_err.sum())
    top_sse = float(top_sq_err.sum())

    return {
        'Subset': subset,
        'Level': level,
        'n': int(len(abs_err)),
        'ExactZero_n': int(exact_zero.sum()),
        'P(error=0)': float(exact_zero.mean()),
        'BoundaryPinnedZero_n': int(boundary_zero.sum()),
        'BoundaryPinnedZero_share': float(boundary_zero.mean()),
        'MAE': float(abs_err.mean()),
        'RMSE': float(np.sqrt(sq_err.mean())),
        'P50AE': float(np.quantile(abs_err, 0.50)),
        'P90AE': float(np.quantile(abs_err, 0.90)),
        'P95AE': float(np.quantile(abs_err, 0.95)),
        'P99AE': float(np.quantile(abs_err, 0.99)),
        'P99.5AE': float(np.quantile(abs_err, 0.995)),
        'MaxAE': float(abs_err.max()),
        'Top1%_n': top_n,
        'Top1%_SSE': top_sse,
        'Top1%_SSE_share': (top_sse / total_sse) if total_sse > 0 else np.nan,
        'Top1%_RMSE': float(np.sqrt(top_sq_err.mean())),
    }


print('\n计算 bin-wise 指标...')
metrics = {(v, m): compute_bin_metrics(y_true, preds[(v, m)])
           for v, m, _ in SERIES}
# v1→v2: 结构特征增益 (逐树模型); v2→v3: ALIGNN 相对各树模型增益
delta    = {m: compute_delta(metrics[('v1', m)], metrics[('v2', m)]) for m in MODELS}
delta_v3 = {m: compute_delta(metrics[('v2', m)], metrics[('v3', 'gnn')]) for m in MODELS}

distribution_rows = []
for subset, mask in DISTRIBUTION_SUBSETS:
    for level, key in DISTRIBUTION_SERIES:
        distribution_rows.append(
            compute_distribution_row(y_true[mask], preds[key][mask], subset, level)
        )
distribution_summary = pd.DataFrame(distribution_rows)


def reduction_factor(old, new):
    """返回 old/new; new 为 0 时保留数学含义, 避免静默除零。"""
    if new > 0:
        return old / new
    if old > 0:
        return np.inf
    return np.nan


endpoint_rows = []
for subset, _ in DISTRIBUTION_SUBSETS:
    sub = distribution_summary[distribution_summary['Subset'] == subset].set_index('Level')
    l1 = sub.loc['Level 1 (XGB)']
    l3 = sub.loc['Level 3 (ALIGNN)']
    endpoint_rows.append({
        'Subset': subset,
        'n': int(l1['n']),
        'P50_reduction_factor_L1_to_L3': reduction_factor(l1['P50AE'], l3['P50AE']),
        'MAE_reduction_factor_L1_to_L3': reduction_factor(l1['MAE'], l3['MAE']),
        'P99_change_eV_L3_minus_L1': l3['P99AE'] - l1['P99AE'],
        'P99_change_%_L3_vs_L1': ((l3['P99AE'] / l1['P99AE']) - 1) * 100
                                   if l1['P99AE'] > 0 else np.nan,
        'ExactZero_change_percentage_points':
            (l3['P(error=0)'] - l1['P(error=0)']) * 100,
        'Top1%_SSE_ratio_L3_to_L1': (l3['Top1%_SSE'] / l1['Top1%_SSE'])
                                    if l1['Top1%_SSE'] > 0 else np.nan,
        'Top1%_SSE_share_change_percentage_points':
            (l3['Top1%_SSE_share'] - l1['Top1%_SSE_share']) * 100,
    })
endpoint_comparison = pd.DataFrame(endpoint_rows)

# ============================================================
# 保存 CSV + 终端摘要
# ============================================================
print('\n保存 CSV...')
for m in MODELS:
    combined = (metrics[('v1', m)].add_prefix('v1_')
                .join(metrics[('v2', m)].add_prefix('v2_'))
                .join(metrics[('v3', 'gnn')].add_prefix('v3_')))
    out = os.path.join(OUTPUT_DIR, f'bin_error_table_{m}.csv')
    combined.to_csv(out, float_format='%.4f')
    print(f'  → {out}')

delta_all = pd.concat([delta['rf'].add_prefix('RF_v1v2_'),
                       delta['xgb'].add_prefix('XGB_v1v2_'),
                       delta_v3['rf'].add_prefix('RFv2_ALIGNN_'),
                       delta_v3['xgb'].add_prefix('XGBv2_ALIGNN_')], axis=1)
delta_all.to_csv(os.path.join(OUTPUT_DIR, 'bin_error_delta.csv'), float_format='%.4f')
print(f"  → {os.path.join(OUTPUT_DIR, 'bin_error_delta.csv')}")

distribution_out = os.path.join(OUTPUT_DIR, 'absolute_error_distribution_summary.csv')
distribution_summary.to_csv(distribution_out, index=False, float_format='%.8f')
print(f'  → {distribution_out}')

endpoint_out = os.path.join(OUTPUT_DIR, 'absolute_error_endpoint_comparison.csv')
endpoint_comparison.to_csv(endpoint_out, index=False, float_format='%.8f')
print(f'  → {endpoint_out}')

for m in MODELS:
    print('\n' + '=' * 86)
    print(f'{m.upper()} bin-wise 指标 (v1 → v2 → v3=ALIGNN, 官方五折 test)')
    print('=' * 86)
    s = pd.DataFrame({
        'n':            metrics[('v2', m)]['n'].astype(int),
        'MAE_v1':       metrics[('v1', m)]['MAE'],
        'MAE_v2':       metrics[('v2', m)]['MAE'],
        'MAE_v3':       metrics[('v3', 'gnn')]['MAE'],
        'impr%_v1→v2':  delta[m]['MAE_improvement_%'],
        'impr%_v2→v3':  delta_v3[m]['MAE_improvement_%'],
        'Bias_v2':      metrics[('v2', m)]['Bias'],
        'Bias_v3':      metrics[('v3', 'gnn')]['Bias'],
    })
    print(s.to_string(float_format=lambda x: f'{x:+.4f}'))

print('\n' + '=' * 126)
print('三层 clipped 绝对误差分布 (L1/L2=XGB, L3=ALIGNN)')
print('=' * 126)
distribution_console = distribution_summary[
    ['Subset', 'Level', 'n', 'ExactZero_n', 'P(error=0)',
     'MAE', 'P50AE', 'P90AE', 'P95AE', 'P99AE', 'P99.5AE', 'MaxAE',
     'Top1%_SSE', 'Top1%_SSE_share']
]
print(distribution_console.to_string(
    index=False,
    formatters={
        'P(error=0)': lambda x: f'{x:.2%}',
        'MAE': lambda x: f'{x:.5f}',
        'P50AE': lambda x: f'{x:.5f}',
        'P90AE': lambda x: f'{x:.5f}',
        'P95AE': lambda x: f'{x:.5f}',
        'P99AE': lambda x: f'{x:.5f}',
        'P99.5AE': lambda x: f'{x:.5f}',
        'MaxAE': lambda x: f'{x:.5f}',
        'Top1%_SSE': lambda x: f'{x:.4f}',
        'Top1%_SSE_share': lambda x: f'{x:.2%}',
    }
))

print('\n' + '=' * 126)
print('Level 1 (XGB) → Level 3 (ALIGNN) 端点比较')
print('=' * 126)
print(endpoint_comparison.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
print('\n注: P50/MAE 的改善倍数必须同指标比较; Top 1% SSE share 是相对份额, '
      '需结合 Top1%_SSE 的绝对量解读。')

# 导出代表系列及 RF-v2 误差最大的样本 (含 mbid, 供案例分析)
for ver, m in [('v1', 'xgb'), ('v2', 'rf'), ('v2', 'xgb'), ('v3', 'gnn')]:
    ae = np.abs(y_true - preds[(ver, m)])
    idx = np.argsort(ae)[::-1][:50]
    pd.DataFrame({'mbid': mbids[idx], 'y_true': y_true[idx],
                  f'y_pred_{ver}': preds[(ver, m)][idx], 'abs_err': ae[idx]}
                 ).to_csv(os.path.join(OUTPUT_DIR, f'top50_err_{ver}_{m}.csv'), index=False)

# ============================================================
# 可视化
# ============================================================
print('\n生成图表...')

C_V1_RF, C_V2_RF   = '#888780', '#444441'
C_V1_XGB, C_V2_XGB = '#85B7EB', '#185FA5'
C_V3_ALIGNN        = '#D97706'

def bin_vals(df, col):
    return [df.loc[b, col] if b in df.index else np.nan for b in ALL_BINS]

bin_n = [int(v) for v in bin_vals(metrics[('v3', 'gnn')], 'n')]
xtick_labels = [f'{b}\n(n={n:,})' if n < SMALL_N_WARN else b
                for b, n in zip(ALL_BINS, bin_n)]   # 小样本 bin 在轴标签直接标 n

# ---- 图1: MAE 分bin对比 (5 根柱: v1/v2 × rf/xgb + ALIGNN) ----
fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(ALL_BINS)); w = 0.16
for i, (key, color, label, hatch) in enumerate([
        (('v1', 'rf'),  C_V1_RF,   'Level 1 (RF)',  '//'),
        (('v2', 'rf'),  C_V2_RF,   'Level 2 (RF)',   ''),
        (('v1', 'xgb'), C_V1_XGB,  'Level 1 (XGB)', '//'),
        (('v2', 'xgb'), C_V2_XGB,  'Level 2 (XGB)',  ''),
        (('v3', 'gnn'), C_V3_ALIGNN, 'Level 3 (ALIGNN)', '')]):
    ax.bar(x + (i - 2) * w, bin_vals(metrics[key], 'MAE'), w, label=label,
           color=color, hatch=hatch, edgecolor='white', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(xtick_labels)
ax.set_xlabel('True band gap (eV)'); ax.set_ylabel('MAE (eV)')
ax.set_title('Bin-wise MAE: Level 1 (composition) vs Level 2 (+descriptors) vs Level 3 (ALIGNN) — official 5-fold test')
ax.legend(fontsize=9, ncol=3); ax.grid(axis='y', alpha=0.3)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, 'bin_mae_comparison.png'), dpi=150)
plt.close(fig)

# ---- 图1B: MAE 分bin (只保留各 Level 的 XGB/ALIGNN 代表系列) ----
fig, ax = plt.subplots(figsize=(11, 4.5))
w = 0.25
ax.bar(x - w, bin_vals(metrics[('v1', 'xgb')], 'MAE'), w,
       label='Level 1 (XGB)', color=C_V1_XGB, hatch='//',
       edgecolor='white', linewidth=0.5)
ax.bar(x, bin_vals(metrics[('v2', 'xgb')], 'MAE'), w,
       label='Level 2 (XGB)', color=C_V2_XGB,
       edgecolor='white', linewidth=0.5)
ax.bar(x + w, bin_vals(metrics[('v3', 'gnn')], 'MAE'), w,
       label='Level 3 (ALIGNN)', color=C_V3_ALIGNN,
       edgecolor='white', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(xtick_labels)
ax.set_xlabel('True band gap (eV)'); ax.set_ylabel('MAE (eV)')
ax.set_title('Bin-wise MAE — Level 1 (XGB) vs Level 2 (XGB) vs Level 3 (ALIGNN)')
ax.legend(fontsize=9, ncol=3); ax.grid(axis='y', alpha=0.3)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, 'bin_mae_level_comparison.png'), dpi=150)
plt.close(fig)

# ---- 图2: Bias 分bin (五系列, 与图1 同视觉语言; Level 1 打斜线 hatch) ----
fig, ax = plt.subplots(figsize=(11, 4))
w = 0.16
for i, (key, color, label, hatch) in enumerate([
        (('v1', 'rf'),  C_V1_RF,   'Level 1 (RF)',  '//'),
        (('v2', 'rf'),  C_V2_RF,   'Level 2 (RF)',   ''),
        (('v1', 'xgb'), C_V1_XGB,  'Level 1 (XGB)', '//'),
        (('v2', 'xgb'), C_V2_XGB,  'Level 2 (XGB)',  ''),
        (('v3', 'gnn'), C_V3_ALIGNN, 'Level 3 (ALIGNN)', '')]):
    ax.bar(x + (i - 2) * w, bin_vals(metrics[key], 'Bias'), w, label=label,
           color=color, hatch=hatch, edgecolor='white', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x); ax.set_xticklabels(xtick_labels)
ax.set_xlabel('True band gap (eV)'); ax.set_ylabel('Bias = mean(pred − true) (eV)')
ax.set_title('Bin-wise prediction bias — Level 1 vs Level 2 vs Level 3 (all predictions clipped at 0)')
ax.legend(fontsize=9, ncol=3); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, 'bin_bias_comparison.png'), dpi=150)
plt.close(fig)

# ---- 图2B: Bias 分bin (只保留各 Level 的 XGB/ALIGNN 代表系列) ----
fig, ax = plt.subplots(figsize=(11, 4))
w = 0.25
ax.bar(x - w, bin_vals(metrics[('v1', 'xgb')], 'Bias'), w,
       label='Level 1 (XGB)', color=C_V1_XGB, hatch='//',
       edgecolor='white', linewidth=0.5)
ax.bar(x, bin_vals(metrics[('v2', 'xgb')], 'Bias'), w,
       label='Level 2 (XGB)', color=C_V2_XGB,
       edgecolor='white', linewidth=0.5)
ax.bar(x + w, bin_vals(metrics[('v3', 'gnn')], 'Bias'), w,
       label='Level 3 (ALIGNN)', color=C_V3_ALIGNN,
       edgecolor='white', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x); ax.set_xticklabels(xtick_labels)
ax.set_xlabel('True band gap (eV)'); ax.set_ylabel('Bias = mean(pred − true) (eV)')
ax.set_title('Bin-wise prediction bias — Level 1 vs Level 2 vs Level 3 (all predictions clipped at 0)')
ax.legend(fontsize=9, ncol=3); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, 'bin_bias_level_comparison.png'), dpi=150)
plt.close(fig)

# ---- 图3: 样本数分布 ----
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(ALL_BINS, bin_n, color=C_V2_XGB, edgecolor='white')
ax.set_yscale('log')
ax.set_xlabel('True band gap (eV)'); ax.set_ylabel('Sample count (log)')
ax.set_title('Sample distribution — zero gap (Eg=0) shown separately')
for i, v in enumerate(bin_n):
    ax.text(i, v * 1.1, f'{v:,}', ha='center', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, 'bin_sample_count.png'), dpi=150)
plt.close(fig)

# ---- 图4: 两级增益对比 (XGB 基线): v1→v2 结构特征 vs v2→v3 ALIGNN ----
fig, ax = plt.subplots(figsize=(10, 4.5))
w = 0.35
impr_v1v2 = bin_vals(delta['xgb'],    'MAE_improvement_%')   # 结构特征增益
impr_v2v3 = bin_vals(delta_v3['xgb'], 'MAE_improvement_%')   # ALIGNN 相对 XGB v2 增益
ax.bar(x - w/2, impr_v1v2, w, label='Level 1→2 (+descriptors, XGB)', color=C_V2_XGB, edgecolor='white')
ax.bar(x + w/2, impr_v2v3, w, label='Level 2→3 (ALIGNN vs XGB)',          color=C_V3_ALIGNN, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
# 每组柱子上方标 n
ymax = np.nanmax([impr_v1v2, impr_v2v3])
for xi, n in zip(x, bin_n):
    ax.text(xi, ymax * 1.08, f'n={n:,}', ha='center', fontsize=8, color='#555555')
ax.set_ylim(top=ymax * 1.2)
ax.set_xticks(x); ax.set_xticklabels(xtick_labels)
ax.set_xlabel('True band gap (eV)'); ax.set_ylabel('MAE improvement (%)')
ax.set_title('Two-stage gains: descriptors (Level 1\u21922) vs ALIGNN (Level 2\u21923) — small-n bins not interpreted')
ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, 'bin_mae_improvement.png'), dpi=150)
plt.close(fig)

# ---- 图5: 三层 clipped 绝对误差分位数曲线 (ALL / zero-gap / non-zero-gap) ----
# q 在尾部加密; 横轴仍显示真实 percentile, P99.5 的精确数值见 CSV。
quantile_grid = np.unique(np.concatenate([
    np.linspace(0.0, 0.90, 451),
    np.linspace(0.90, 0.99, 181),
    np.linspace(0.99, 0.999, 91),
    np.array([0.50, 0.90, 0.95, 0.99, 0.995]),
]))
dist_plot_series = [
    ('Level 1 (XGB)', ('v1', 'xgb'), C_V1_XGB),
    ('Level 2 (XGB)', ('v2', 'xgb'), C_V2_XGB),
    ('Level 3 (ALIGNN)', ('v3', 'gnn'), C_V3_ALIGNN),
]
summary_lookup = distribution_summary.set_index(['Subset', 'Level'])

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
for ax, (subset, mask) in zip(axes, DISTRIBUTION_SUBSETS):
    panel_stats = []
    for level, key, color in dist_plot_series:
        abs_err = np.abs(y_true[mask] - preds[key][mask])
        q_values = np.quantile(abs_err, quantile_grid)
        ax.plot(quantile_grid * 100, q_values, color=color, linewidth=2.0,
                label=level, zorder=2)

        stats = summary_lookup.loc[(subset, level)]
        p50, p99 = float(stats['P50AE']), float(stats['P99AE'])
        ax.scatter([50], [p50], color=color, marker='o', s=28, zorder=4)
        ax.scatter([99], [p99], color=color, marker='D', s=27, zorder=4)
        short_level = level.replace('Level ', 'L').replace(' (', '-').replace(')', '')
        panel_stats.append((
            color,
            f'{short_level}: {stats["P(error=0)"]:.1%} / {p50:.4f} / {p99:.4f}'
        ))

    ax.axvline(50, color='#777777', linewidth=0.7, linestyle='--', alpha=0.6)
    ax.axvline(99, color='#777777', linewidth=0.7, linestyle=':', alpha=0.8)
    ax.set_yscale('symlog', linthresh=QUANTILE_SYMLOG_LINTHRESH, linscale=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 99.9)
    ax.set_xticks([0, 25, 50, 75, 90, 95, 99])
    ax.set_xlabel('Percentile of absolute error')
    ax.set_title(f'{subset}\n(n={int(mask.sum()):,})', fontsize=10.5)
    ax.grid(alpha=0.22, which='both')
    ax.text(0.02, 0.98, 'P(error=0) / P50 / P99', transform=ax.transAxes,
            va='top', ha='left', fontsize=7.5, color='#333333')
    for line_i, (color, line) in enumerate(panel_stats):
        ax.text(0.02, 0.915 - 0.065 * line_i, line, transform=ax.transAxes,
                va='top', ha='left', fontsize=7.5, color=color)

axes[0].set_ylabel('Absolute error |prediction − truth| (eV, symlog)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.985))
fig.suptitle('Clipped absolute-error quantiles — official pooled 5-fold test\n'
             'Circles mark P50; diamonds mark P99; exact-zero rates are post-clipping',
             y=1.08, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(OUTPUT_DIR, 'absolute_error_quantile_curves.png'),
            dpi=180, bbox_inches='tight')
plt.close(fig)

print(f'\n✅ 全部完成, 输出目录: {os.path.abspath(OUTPUT_DIR)}')
