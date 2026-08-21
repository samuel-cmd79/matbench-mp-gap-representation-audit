"""
八面体-过渡金属混杂自查
========================
问题: SHAP 显示 mean octahedral CN_6 高 → 带隙预测低。
      但八面体配位在过渡金属化合物中富集, 而过渡金属本身(d 态)压低带隙。
      若高八面体组的 TM 占比显著更高, 则该方向不能单独归因于配位几何。

做法:
  1. 从 matbench 官方 task 取全部样本 (五折 test 拼接 = 全数据集);
  2. 逐样本: 是否含过渡金属 (pymatgen Element.is_transition_metal);
  3. 读特征缓存中的 SiteStatsFingerprint 列 'mean octahedral CN_6';
  4. 按该特征四分位分组, 报告各组: TM 占比 / 标签均值;
  5. 分层对照: 仅 TM 样本内 与 仅非 TM 样本内, 分别看
     高八面体 (>中位数) vs 低八面体 (<=中位数) 的标签均值差。
     若分层后方向仍在, 说明配位几何有独立信号; 若消失, 则纯混杂。

用法: 改 CACHE_PATTERN 指向你的特征缓存目录后运行。
      缓存文件名按你日志里的格式: fold_{fold}_test_SiteStatsFingerprint.pkl
"""

import pickle
import numpy as np
import pandas as pd

# ================= CONFIG =================
CACHE_PATTERN = '../matbench_cache/fold_{fold}_test_SiteStatsFingerprint.pkl'
OCT_COL = 'mean octahedral CN_6'   # 若列名不同, 先打印 df.columns 核对
FOLDS = [0, 1, 2, 3, 4]
TASK_NAME = 'matbench_mp_gap'
# ==========================================

from matbench.bench import MatbenchBenchmark
from pymatgen.core.periodic_table import Element

mb = MatbenchBenchmark(autoload=False)
task = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
task.load()
target_col = task.metadata['target']

rows = []
for fold in FOLDS:
    df = task.get_test_data(fold, as_type='df', include_target=True)

    with open(CACHE_PATTERN.format(fold=fold), 'rb') as f:
        feat = pickle.load(f)
    if OCT_COL not in feat.columns:
        cand = [c for c in feat.columns if 'octahedral' in c]
        raise KeyError(f'找不到列 {OCT_COL!r}; 含 octahedral 的候选列: {cand}')
    if len(feat) != len(df):
        raise ValueError(f'fold {fold}: 缓存 {len(feat)} 行 != 官方 test {len(df)} 行')

    oct_vals = feat[OCT_COL].to_numpy()
    for (mbid, r), oc in zip(df.iterrows(), oct_vals):
        comp = r['structure'].composition
        has_tm = any(Element(el.symbol).is_transition_metal for el in comp.elements)
        rows.append({'mbid': str(mbid), 'fold': fold, 'y': float(r[target_col]),
                     'oct': float(oc) if np.isfinite(oc) else np.nan,
                     'has_tm': has_tm})

d = pd.DataFrame(rows)
n_nan = int(d['oct'].isna().sum())
d = d.dropna(subset=['oct'])
print(f'样本: {len(d)} (剔除特征缺失 {n_nan})  TM 占比全体: {d["has_tm"].mean():.1%}\n')

# ---- 1) 四分位分组 ----
# 大量样本 oct=0 (无八面体位点), 四分位可能退化; 先看零值占比
zero_share = (d['oct'] == 0).mean()
print(f'oct=0 的样本占比: {zero_share:.1%}')
if zero_share > 0.5:
    print('→ 零值过半, 改用三组: oct=0 / 0<oct<=中位数(正值) / oct>中位数(正值)')
    pos = d[d['oct'] > 0]
    med = pos['oct'].median()
    d['grp'] = np.where(d['oct'] == 0, 'oct = 0',
               np.where(d['oct'] <= med, 'oct low(+)', 'oct high(+)'))
    order = ['oct = 0', 'oct low(+)', 'oct high(+)']
else:
    d['grp'] = pd.qcut(d['oct'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    order = list(d['grp'].cat.categories)

summ = d.groupby('grp').agg(n=('y', 'size'),
                            tm_share=('has_tm', 'mean'),
                            mean_gap=('y', 'mean'),
                            median_gap=('y', 'median')).reindex(order)
summ['tm_share'] = (summ['tm_share'] * 100).round(1)
print('\n按 mean octahedral CN_6 分组:')
print(summ.to_string(float_format=lambda x: f'{x:.3f}'))

# ---- 2) 分层对照 ----
print('\n分层对照 (各层内: 高八面体 = 该层正值中位数以上):')
for name, sub in [('含 TM', d[d['has_tm']]), ('不含 TM', d[~d['has_tm']])]:
    pos = sub[sub['oct'] > 0]
    if len(pos) < 100:
        print(f'  {name}: 正值样本过少 ({len(pos)}), 跳过')
        continue
    med = pos['oct'].median()
    hi = sub[sub['oct'] > med]
    lo = sub[sub['oct'] <= med]
    print(f'  {name}: n={len(sub)}  '
          f'高八面体 mean gap={hi["y"].mean():.3f} (n={len(hi)})  '
          f'低/零八面体 mean gap={lo["y"].mean():.3f} (n={len(lo)})  '
          f'差={hi["y"].mean() - lo["y"].mean():+.3f} eV')

print("""
判读:
  - 若高八面体组 TM 占比明显高于低组 → 存在混杂, 正文不得将
    '八面体→窄隙' 单独归因于配位几何;
  - 分层对照: 若 TM 层内和非 TM 层内高八面体仍对应更低 gap →
    配位几何有独立于 TM 的信号 (正文可作合并表述);
    若分层后差值接近 0 或翻转 → 纯混杂, 删去该方向或仅列不解读。
""")
