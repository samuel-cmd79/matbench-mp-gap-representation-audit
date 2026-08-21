"""
XGB 训练汇总表 (SI) — 从 training_info json 生成, 替代/伴随训练曲线
====================================================================
数据源: outputs_v1_run0709/xgb_training_info_fold_{1..5}.json
  (注意: 这批文件 fold 是 1-indexed; 其余分析脚本均 0-indexed, 表中两列并列标明)

输出: 终端表 + xgb_training_summary.csv
支撑的 SI 表述: "早停 (patience=200) 在 8000 轮预算内未触发,
final validation MAE 即 best" — 表中 triggered 列与 best==final 校验直接证明。
"""

import json
import pandas as pd
from pathlib import Path

# ================= CONFIG =================
INFO_PATTERN = '../outputs_v1_run0709/xgb_training_info_fold_{fold}.json'
FOLDS_1IDX   = [1, 2, 3, 4, 5]      # 文件名里的 1-indexed fold
OUT_CSV      = './xgb_training_summary.csv'
# ==========================================

rows = []
for f1 in FOLDS_1IDX:
    p = Path(INFO_PATTERN.format(fold=f1))
    if not p.exists():
        print(f'⚠️ 缺 {p}, 跳过')
        continue
    d = json.load(open(p))
    total = d['total_rounds_evaluated']
    best  = d['best_iteration']
    # 早停触发判定: best_iteration 落在最后一轮 (0-indexed) 之前
    # 且 total < 预算上限时才可能是早停; best == total-1 → 跑满预算未触发
    triggered = best < total - 1
    rows.append({
        'fold (file, 1-idx)': f1,
        'fold (analysis, 0-idx)': f1 - 1,
        'total_rounds': total,
        'best_iteration': best,
        'best_val_MAE': d['best_validation_mae'],
        'final_val_MAE': d['final_val_mae'],
        'early_stop_patience': d['early_stopping_rounds'],
        'early_stop_triggered': triggered,
        'final_equals_best': abs(d['final_val_mae'] - d['best_validation_mae']) < 1e-12,
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
df.to_csv(OUT_CSV, index=False, float_format='%.6f')
print(f'\n→ {OUT_CSV}')

if len(df):
    all_no_stop = (~df['early_stop_triggered']).all()
    print(f'\n早停触发情况: {"五折均未触发 (best 均在最后一轮)" if all_no_stop else "部分折触发, 见表"}')
    print(f'best_val_MAE 五折: {df["best_val_MAE"].mean():.4f} ± {df["best_val_MAE"].std():.4f}')
    if all_no_stop:
        print('\n措辞提醒: "未触发早停" 同时意味着验证 MAE 在 8000 轮预算末仍在改善,')
        print('训练由预算而非收敛终止。SI 建议写 "early stopping (patience=200) was not')
        print('triggered within the 8,000-round budget", 并对照旧曲线 PNG 确认末段已趋平——')
        print('若末段仍明显下降, 审稿人可能问预算是否充足, 提前想好回应 (如: 末段斜率')
        print('已 < X eV/千轮, 继续训练收益可忽略)。')
