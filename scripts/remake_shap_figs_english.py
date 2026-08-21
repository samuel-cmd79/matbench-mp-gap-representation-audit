"""
重制 v2 SHAP 蜂群图 + 重要性图 (英文版, 不重训)
================================================
数据源 (全部为 run0709 留档资产):
  - SHAP 值:   matbench_outputs_v2_run0709/shap_values_{model}_fold_{fold}.npy
  - 特征名:    matbench_outputs_v2_run0709/shap_feature_names_{model}.json
  - X_shap 重建: matbench_cache/fold_{fold}_train_{Featurizer}.pkl × 8

X_shap 重建链 (与冻结训练脚本 mp_gap_baseline_v2_run0709.py 逐步对应):
  1. 8 个 featurizer train 缓存 concat (缓存已剔除 structure/comp 列)
  2. 去重列 → get_dummies(非数值列, dummy_na=True) → bool→int
     → to_numeric(coerce) → inf→NaN            [= extract_features 后处理]
  3. 按 feature_names json 重排列序, 缺列补 0    [= align(join='outer') 的效果,
     以 json 为权威, 免受 pandas 版本 Index.union 行为影响]
  4. col_means = X.mean(); X.fillna(col_means).fillna(0)   [= 主循环 427 行]
  5. RandomState(42).choice(len(X), 3000, replace=False)   [= run_shap_analysis 307 行]

校验锚点 (信仰式复现 → 验证式复现):
  a. npy shape == (3000, len(feature_names))
  b. 重排后列集合与 json 完全一致 (缺/多列会打印)
  c. 终端打印 mean|SHAP| top-10 → 与旧 PNG 的排名肉眼比对
  d. 新旧蜂群图并排看点云形状

仅适用于 v2 (8 个 featurizer 是 v2 管线); v1 的蜂群图要重制需 v1 训练脚本的
特征拼接段, 另议。
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# ================= CONFIG =================
CACHE_DIR = Path('../matbench_cache')
V2_OUT    = Path('../matbench_outputs_v2_run0709')
MODEL     = 'xgb'
FOLD      = 0
N_SHAP    = 3000
SEED      = 42
MAX_DISPLAY = 20
OUT_DIR   = Path('./shap_english')
TITLE_BEE = f'SHAP beeswarm — Level 2 (+descriptors, XGB), fold {FOLD}'
TITLE_BAR = f'SHAP feature importance — Level 2 (+descriptors, XGB), fold {FOLD}'
# 冻结脚本中的 featurizer 名 (拼接顺序; 最终列序以 json 为准, 此处顺序不影响结果)
FEATURIZERS = ['DensityFeatures', 'GlobalSymmetryFeatures', 'StructuralHeterogeneity',
               'ChemicalOrdering', 'Dimensionality', 'SiteStatsFingerprint',
               'ElementProperty', 'BandCenter']
# ==========================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 载入 SHAP 值与特征名 ----------
sv_path    = V2_OUT / f'shap_values_{MODEL}_fold_{FOLD}.npy'
names_path = V2_OUT / f'shap_feature_names_{MODEL}.json'
shap_values = np.load(sv_path)
names = [str(n) for n in json.load(open(names_path, encoding='utf-8'))]
print(f'SHAP 值: {shap_values.shape}, 特征名: {len(names)}')
if shap_values.ndim != 2 or shap_values.shape[1] != len(names):
    raise ValueError(f'shape 不符: npy {shap_values.shape} vs 特征名 {len(names)} — '
                     f'确认 npy/json 是同一 run 同一模型')
if shap_values.shape[0] != N_SHAP:
    print(f'⚠️ npy 行数 {shap_values.shape[0]} != N_SHAP={N_SHAP} — '
          f'说明当时训练集不足 {N_SHAP} 行 (n_shap=min 逻辑), 已按实际行数处理')

# ---------- 重建 X_train (步骤 1-2) ----------
print('\n重建 fold train 特征...')
parts = []
for name in FEATURIZERS:
    p = CACHE_DIR / f'fold_{FOLD}_train_{name}.pkl'
    if not p.exists():
        raise FileNotFoundError(f'缺缓存: {p}')
    part = pd.read_pickle(p)
    print(f'  {name}: {part.shape}')
    parts.append(part)
X = pd.concat(parts, axis=1)
X = X.loc[:, ~X.columns.duplicated()].copy()
non_numeric = X.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()
if non_numeric:
    print(f'  非数值列 one-hot: {non_numeric}')
    X = pd.get_dummies(X, columns=non_numeric, dummy_na=True)
bool_cols = X.select_dtypes(include=['bool']).columns
if len(bool_cols):
    X[bool_cols] = X[bool_cols].astype(int)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)
print(f'  重建后: {X.shape}')

# ---------- 按 json 权威列序重排 (步骤 3) ----------
X.columns = [str(c) for c in X.columns]
missing = [c for c in names if c not in X.columns]     # test-only 列 → 补 0 (原 align 行为)
extra   = [c for c in X.columns if c not in set(names)]
if missing:
    print(f'  按 json 补 0 列 {len(missing)} 个 (原 align outer 中来自 test 的列): '
          f'{missing[:5]}{" ..." if len(missing) > 5 else ""}')
if extra:
    print(f'  ⚠️ 重建多出 {len(extra)} 列 (json 中没有, 将丢弃): {extra[:5]} — '
          f'若数量多, 说明库版本导致特征列变了, 排名比对必须过关才可用')
X = X.reindex(columns=names, fill_value=0)

# ---------- 填充 (步骤 4) 与抽样 (步骤 5) ----------
col_means = X.mean(numeric_only=True)
X = X.fillna(col_means).fillna(0)
n_shap = min(N_SHAP, len(X))
rng = np.random.RandomState(SEED)
idx = rng.choice(len(X), n_shap, replace=False)
X_shap = X.iloc[idx]
print(f'\n抽样: 从 {len(X)} 行抽 {n_shap} 行 (RandomState({SEED}).choice, 与冻结脚本同调用)')
if len(X_shap) != shap_values.shape[0]:
    raise ValueError(f'抽样行数 {len(X_shap)} != SHAP 行数 {shap_values.shape[0]}')

# ---------- 校验锚点 c: top-10 排名 ----------
imp = np.abs(shap_values).mean(axis=0)
order = np.argsort(imp)[::-1]
print('\nmean|SHAP| top-10 (与旧 PNG 排名肉眼比对, 必须一致):')
for r, i in enumerate(order[:10], 1):
    print(f'  {r:>2}. {names[i]}  ({imp[i]:.4f})')

# ---------- 出图 ----------
plt.figure()
shap.summary_plot(shap_values, X_shap, feature_names=names,
                  show=False, max_display=MAX_DISPLAY)
plt.title(TITLE_BEE)
plt.tight_layout()
bee_path = OUT_DIR / f'shap_beeswarm_{MODEL}_fold_{FOLD}_en.png'
plt.savefig(bee_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'\n→ {bee_path}')

plt.figure()
shap.summary_plot(shap_values, X_shap, feature_names=names,
                  plot_type='bar', show=False, max_display=MAX_DISPLAY)
plt.title(TITLE_BAR)
plt.tight_layout()
bar_path = OUT_DIR / f'shap_importance_{MODEL}_fold_{FOLD}_en.png'
plt.savefig(bar_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'→ {bar_path}')

print('\n最后一步 (人工): 新旧蜂群图并排比对点云形状; top-10 排名已在上方打印。')
print('两者都对上 → X_shap 复现成立, 图可入论文。')
