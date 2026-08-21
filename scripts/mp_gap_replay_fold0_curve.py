# %% [markdown]
# # 材料带隙预测 - 本地运行版本
# 
# **整体流程：**
# 1. 导入库
# 2. 配置参数（在这里改 TEST_MODE、选模型）
# 3. 加载 matbench 数据
# 4. 定义特征提取函数
# 5. 定义 GNN 数据导出函数
# 6. 定义模型训练函数
# 7. 定义 SHAP 分析函数
# 8. **运行主流程**（从这里开始真正跑）
# 
# > 💡 **新手提示**：Cell 1~7 是「准备工具」，Cell 8 是「开始干活」。
# > 每次打开 notebook，需要从头到尾按顺序跑一遍所有 cell。

# %%
# ==================== Cell 1：导入所有需要的库 ====================
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from matbench.bench import MatbenchBenchmark
from matminer.featurizers.composition import ElementProperty, BandCenter
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure import (
	DensityFeatures,
	GlobalSymmetryFeatures,
	StructuralHeterogeneity,
	ChemicalOrdering,
	Dimensionality,
)
from matminer.featurizers.structure.sites import SiteStatsFingerprint

import xgboost as xgb
import shap

print('✅ 所有库导入成功!')

# %%
# ==================== Cell 2：配置参数 ====================
# ★ 这是你最常改的地方 ★
#
# 想快速测试代码是否能跑：TEST_MODE = True
# 想正式跑全量数据拿官方分数：TEST_MODE = False
#
# 想跑 XGBoost：MODELS_TO_RUN = ["xgb"]
# 想跑随机森林：MODELS_TO_RUN = ["rf"]
# 注意：每次只跑一个，不要同时放两个

MODELS_TO_RUN = ["xgb"]   # ← 重放对象是 xgb fold 0（冻结产物在 predictions_xgb/）

TASK_NAME  = "matbench_mp_gap"
TEST_MODE  = False     # ← 改这里切换模式
TEST_N     = 50      # 测试模式下每个 fold 用多少条数据
TEST_FOLDS = 1       # 测试模式下跑几个 fold
N_JOBS     = 6       # 并行数，Mac 建议先用 1

SEED     = 42
N_SPLITS = 5
TOPK     = 50

# 本地存储目录（会自动创建）
# ★ 重放护栏：输出目录用全新的，物理隔离，任何写盘都不可能覆盖冻结产物 ★
OUT_DIR        = Path("../replay_fold0_curve")           # 本次重放的输出（全新目录）
FROZEN_DIR     = Path("../matbench_outputs_v2_run0709")  # 冻结版本，只读对账，绝不写入
CACHE_DIR      = Path("../matbench_cache")
GNN_EXPORT_DIR = Path("../gnn_export")

# ★ 重放开关：只跑列表里的 fold；设为 None 恢复跑全部 ★
FOLDS_OVERRIDE = [0]

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GNN_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CACHE    = CACHE_DIR / "features_cache_structural.pkl"
FORCE_REFEATURIZE = False 
FORCE_RETRAIN     = True

print(f'✅ 配置完成')
print(f'   模型: {MODELS_TO_RUN}')
print(f'   TEST_MODE: {TEST_MODE}')
print(f'   输出目录: {OUT_DIR}')

# %%
# ==================== Cell 3：加载 matbench 数据 ====================
# 第一次运行会自动下载数据集，之后从本地缓存读取

mb         = MatbenchBenchmark(autoload=False)
task       = next(t for t in mb.tasks if t.dataset_name == TASK_NAME)
task.load()
target_col = task.metadata["target"]
print(f'✅ 任务加载成功: {TASK_NAME}')
print(f'   预测目标: {target_col}（带隙，单位 eV）')
print(f'   fold 列表: {list(task.folds)}')

# %%
# ==================== Cell 4：定义特征提取函数 ====================
# 把晶体结构（Structure 对象）转换成 XGBoost 能理解的数值特征

def _run_one_featurizer(featurizer, df, col, cache_path):
	"""运行单个特征提取器，优先读缓存"""
	if cache_path.exists() and not FORCE_REFEATURIZE:
		try:
			cached = pd.read_pickle(cache_path)
			if len(cached) == len(df):
				print(f'    ✅ 缓存命中: {cache_path.name}')
				return cached
			print(f'    ⚠️ 缓存长度不匹配，重新提取: {cache_path.name}')
		except Exception as e:
			print(f'    ⚠️ 缓存读取失败: {e}')

	print(f'    ⏳ 提取中: {cache_path.name}')
	result_df = featurizer.featurize_dataframe(df.copy(), col, ignore_errors=True)
	result_df = result_df.drop(columns=[col], errors='ignore')
	pd.to_pickle(result_df, cache_path)
	print(f'    💾 已缓存: {cache_path.name}')
	return result_df


def extract_features(structures, split_name: str) -> pd.DataFrame:
	"""提取全部特征（结构特征 + 化学式特征）"""
	print(f'\n  [特征提取] {split_name}')

	struct_df = pd.DataFrame({'structure': structures})
	comp_df   = pd.DataFrame({'comp': [s.composition for s in structures]})

	struct_featurizer_list = [
		('DensityFeatures',         DensityFeatures()),
		('GlobalSymmetryFeatures',  GlobalSymmetryFeatures()),
		('StructuralHeterogeneity', StructuralHeterogeneity()),
		('ChemicalOrdering',        ChemicalOrdering()),
		('Dimensionality',          Dimensionality()),
		('SiteStatsFingerprint',    SiteStatsFingerprint.from_preset('CrystalNNFingerprint_ops')),
	]
	comp_featurizer_list = [
		('ElementProperty', ElementProperty.from_preset('magpie')),
		('BandCenter',      BandCenter()),
	]

	all_parts = []
	print('  [1/2] 结构特征...')
	for name, fzer in struct_featurizer_list:
		fzer.set_n_jobs(N_JOBS)
		part = _run_one_featurizer(fzer, struct_df, 'structure', CACHE_DIR / f'{split_name}_{name}.pkl')
		all_parts.append(part)

	print('  [2/2] 化学式特征...')
	for name, fzer in comp_featurizer_list:
		fzer.set_n_jobs(N_JOBS)
		part = _run_one_featurizer(fzer, comp_df, 'comp', CACHE_DIR / f'{split_name}_{name}.pkl')
		all_parts.append(part)

	X = pd.concat(all_parts, axis=1)

	# 防止不同 featurizer 产生重名列
	X = X.loc[:, ~X.columns.duplicated()].copy()

	# 找出非数值列，比如 crystal_system 这种字符串类别
	non_numeric_cols = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

	if non_numeric_cols:
		print(f"  ⚠️ 发现非数值特征列，将进行 one-hot 编码: {non_numeric_cols}")

		# 把字符串/类别列转成 one-hot 数值列
		X = pd.get_dummies(X, columns=non_numeric_cols, dummy_na=True)

	# 把 bool 转成 0/1
	bool_cols = X.select_dtypes(include=["bool"]).columns
	if len(bool_cols) > 0:
		X[bool_cols] = X[bool_cols].astype(int)

	# 全部强制转成数值，转不了的变 NaN
	X = X.apply(pd.to_numeric, errors="coerce")

	# 处理 inf；NaN 保留，填充延后到 fold 内用训练集统计量执行 (fit on train)
	X = X.replace([np.inf, -np.inf], np.nan)

	print(f"  ✅ {split_name} 特征提取完成，总特征数: {X.shape[1]}")
	return X

# %%
# ==================== Cell 5：定义 GNN 数据导出函数 ====================
# 把每个 fold 的原始 Structure + 标签存成 pkl，以后 GNN 直接读

def export_fold_for_gnn(task_local, fold: int):
	"""导出单个 fold 的原始数据供 GNN 使用"""
	fold_dir = GNN_EXPORT_DIR / f'fold_{fold}'
	fold_dir.mkdir(parents=True, exist_ok=True)

	if (fold_dir / 'train_inputs.pkl').exists():
		print(f'  ✅ GNN 数据已存在，跳过: fold_{fold}')
		return

	print(f'  📦 导出 GNN 数据: fold_{fold}')
	train_df      = task_local.get_train_and_val_data(fold, as_type='df')
	train_df      = train_df.rename(columns=lambda x: x.strip())
	train_inputs  = train_df['structure'].tolist()
	train_outputs = train_df[target_col].tolist()

	test_df     = task_local.get_test_data(fold, as_type='df')
	test_df     = test_df.rename(columns=lambda x: x.strip())
	test_inputs = test_df['structure'].tolist()
	test_ids    = list(test_df.index)

	with open(fold_dir / 'train_inputs.pkl',  'wb') as f: pickle.dump(train_inputs,  f)
	with open(fold_dir / 'train_outputs.pkl', 'wb') as f: pickle.dump(train_outputs, f)
	with open(fold_dir / 'test_inputs.pkl',   'wb') as f: pickle.dump(test_inputs,   f)
	with open(fold_dir / 'test_ids.pkl',      'wb') as f: pickle.dump(test_ids,      f)

	print(f'  ✅ fold_{fold} 导出完成: 训练 {len(train_inputs)} 条 / 测试 {len(test_inputs)} 条')

print('✅ GNN 导出函数定义完成')

# %%
# ==================== Cell 6：定义模型训练函数 ====================

def make_model(model_name: str):
	if model_name == 'rf':
		return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=6)
	elif model_name == 'xgb':
		return None
	else:
		raise ValueError(f'未知模型: {model_name}')


def xgb_train_and_predict(X_train, y_train, X_test, fold_idx: int):
	"""训练 XGBoost（带早停）并预测测试集"""
	X_tr, X_val, y_tr, y_val = train_test_split(
		X_train, y_train, test_size=0.2, random_state=SEED, shuffle=True
	)
	dtr   = xgb.DMatrix(X_tr,   label=y_tr)
	dval  = xgb.DMatrix(X_val,  label=y_val)
	dtest = xgb.DMatrix(X_test)

	params = {
		'objective':        'reg:squarederror',
		'eval_metric':      'mae',
		'eta':              0.03,
		'max_depth':        6,
		'min_child_weight': 5,
		'subsample':        0.8,
		'colsample_bytree': 0.8,
		'lambda':           1.0,
		'alpha':            1e-3,
		'tree_method':      'hist',
		'seed':             SEED,
	}

	evals_result = {}
	print(f'  [XGBoost Fold {fold_idx+1}] 开始训练...')

	num_boost_round       = 500  if TEST_MODE else 8000
	early_stopping_rounds = 50   if TEST_MODE else 200

	booster = xgb.train(
		params=params, dtrain=dtr,
		num_boost_round=num_boost_round,
		evals=[(dval, 'validation')],
		evals_result=evals_result,
		early_stopping_rounds=early_stopping_rounds,
		verbose_eval=50,
	)

	best_it    = booster.best_iteration
	best_score = booster.best_score
	print(f'  [XGBoost Fold {fold_idx+1}] 完成: 最优轮数={best_it}, 验证MAE={best_score:.4f} eV')

	try:
		y_pred = booster.predict(dtest, iteration_range=(0, best_it + 1))
	except TypeError:
		y_pred = booster.predict(dtest, ntree_limit=best_it + 1)

	# 保存训练曲线
	if 'validation' in evals_result and 'mae' in evals_result['validation']:
		val_mae_list = evals_result['validation']['mae']
		epochs = list(range(1, len(val_mae_list) + 1))
		plt.figure(figsize=(10, 6))
		plt.plot(epochs, val_mae_list, label='Validation MAE', color='blue')
		plt.axvline(x=best_it+1, color='red', linestyle='--',
					label=f'Best iteration: {best_it} (MAE={best_score:.4f} eV)')
		plt.xlabel('Boosting round'); plt.ylabel('MAE (eV)')
		plt.title(f'Level 2 xgb fold {fold_idx}')
		plt.legend(); plt.grid(True, alpha=0.3)
		curve_path = OUT_DIR / f'xgb_mae_curve_fold_{fold_idx}.png'
		plt.savefig(curve_path, dpi=150, bbox_inches='tight')
		plt.show()
		print(f'  → 训练曲线已保存: {curve_path}')

	# ★ 改动一：落盘完整验证曲线（写入全新 OUT_DIR，不触碰冻结目录）★
	with open(OUT_DIR / f'evals_result_fold_{fold_idx}.json', 'w') as f:
		json.dump({'validation_mae': evals_result['validation']['mae'],
				   'best_iteration': int(booster.best_iteration),
				   'best_score': float(booster.best_score)}, f)
	print(f'  → 验证曲线 JSON 已保存: {OUT_DIR / f"evals_result_fold_{fold_idx}.json"}')

	return booster, y_pred

print('✅ 模型训练函数定义完成')

# %%
# ==================== Cell 7：定义 SHAP 和特征重要性函数 ====================

def run_shap_analysis(booster_or_model, X_train, feature_names, fold_idx: int, model_name: str):
	"""计算 SHAP 值，保存蜂群图和重要性条形图"""
	print(f'\n  [SHAP 分析] Fold {fold_idx+1}...')

	n_shap  = min(3000, len(X_train))
	rng     = np.random.RandomState(SEED)
	idx     = rng.choice(len(X_train), n_shap, replace=False)
	X_shap  = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
	explainer   = shap.TreeExplainer(booster_or_model)
	shap_values = explainer.shap_values(X_shap, check_additivity=False)
	
	# 蜂群图
	plt.figure(figsize=(12, 8))
	shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False, max_display=20)
	plt.title(f'SHAP 蜂群图 - {model_name} fold {fold_idx}')
	plt.tight_layout()
	beeswarm_path = OUT_DIR / f'shap_beeswarm_{model_name}_fold_{fold_idx}.png'
	plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
	plt.show()
	print(f'  → SHAP 蜂群图已保存: {beeswarm_path}')

	# 重要性条形图
	plt.figure(figsize=(12, 8))
	shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
					  plot_type='bar', show=False, max_display=20)
	plt.title(f'SHAP 特征重要性 - {model_name} fold {fold_idx}')
	plt.tight_layout()
	bar_path = OUT_DIR / f'shap_importance_{model_name}_fold_{fold_idx}.png'
	plt.savefig(bar_path, dpi=150, bbox_inches='tight')
	plt.show()
	print(f'  → SHAP 重要性图已保存: {bar_path}')

	np.save(OUT_DIR / f'shap_values_{model_name}_fold_{fold_idx}.npy', shap_values)
	with open(OUT_DIR / f'shap_feature_names_{model_name}.json', 'w', encoding='utf-8') as f:
		json.dump(list(feature_names), f, ensure_ascii=False)


def plot_feature_importance(booster_or_model, feature_names, fold_idx: int, model_name: str):
	"""画模型内置特征重要性图"""
	print(f'\n  [特征重要性] Fold {fold_idx+1}...')

	if model_name == 'xgb':
		importance_dict = booster_or_model.get_score(importance_type='gain')
		importances = np.array([importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))])
	else:
		importances = booster_or_model.feature_importances_

	top_k   = 20
	indices = np.argsort(importances)[::-1][:top_k]

	plt.figure(figsize=(12, 8))
	plt.barh([feature_names[i] for i in indices][::-1], importances[indices][::-1], color='steelblue')
	plt.xlabel('特征重要性 (Gain)')
	plt.title(f'Top {top_k} 特征重要性 - {model_name} fold {fold_idx}')
	plt.tight_layout()
	fig_path = OUT_DIR / f'feature_importance_{model_name}_fold_{fold_idx}.png'
	plt.savefig(fig_path, dpi=150, bbox_inches='tight')
	plt.show()
	print(f'  → 特征重要性图已保存: {fig_path}')

print('✅ SHAP 和特征重要性函数定义完成')

# %%
# ==================== Cell 8：定义主函数 ====================

def run_one_model(model_name: str):
	"""完整运行一个模型的所有 fold"""
	predictions_dir = OUT_DIR / f'predictions_{model_name}'
	predictions_dir.mkdir(parents=True, exist_ok=True)

	mb_local   = MatbenchBenchmark(autoload=False)
	task_local = next(t for t in mb_local.tasks if t.dataset_name == TASK_NAME)
	task_local.load()

	folds_to_run = list(task_local.folds)
	# ★ 改动二：只重放指定 fold（默认 [0]）★
	if FOLDS_OVERRIDE is not None:
		folds_to_run = [f for f in folds_to_run if f in FOLDS_OVERRIDE]
		print(f'⚠️  FOLDS_OVERRIDE={FOLDS_OVERRIDE}：只跑 fold {folds_to_run}')
	if TEST_MODE:
		folds_to_run = folds_to_run[:TEST_FOLDS]
		print(f'⚠️  TEST_MODE=True：只跑前 {len(folds_to_run)} 个 fold，每个取前 {TEST_N} 条')

	for fold_idx, fold in enumerate(folds_to_run):
		print(f'\n{"="*60}')
		print(f'[{model_name}] Fold {fold_idx+1}/{len(folds_to_run)}')
		print(f'{"="*60}')

		pred_suffix = f'test{TEST_N}_fold_{fold}' if TEST_MODE else f'fold_{fold}'
		pred_file   = predictions_dir / f'pred_{pred_suffix}.npy'
		if pred_file.exists() and not FORCE_RETRAIN:
			print(f'  预测文件已存在，跳过: {pred_file}')
			y_pred = np.load(pred_file)
			if not TEST_MODE:
				task_local.record(fold, y_pred)
			continue

		# 获取数据
		train_df = task_local.get_train_and_val_data(fold, as_type='df')
		test_df  = task_local.get_test_data(fold, as_type='df')
		train_df = train_df.rename(columns=lambda x: x.strip())
		test_df  = test_df.rename(columns=lambda x: x.strip())

		if TEST_MODE:
			train_df = train_df.iloc[:TEST_N].copy()
			test_df  = test_df.iloc[:TEST_N].copy()

		X_train_raw = train_df['structure'].tolist()
		y_train     = train_df[target_col].to_numpy()
		X_test_raw  = test_df['structure'].tolist()
		print(f'  训练集: {len(X_train_raw)} 条，测试集: {len(X_test_raw)} 条')

		# 导出 GNN 数据（只在全量模式下）
		if not TEST_MODE:
			export_fold_for_gnn(task_local, fold)

		# 提取特征
		cache_prefix = f"test{TEST_N}_fold_{fold}" if TEST_MODE else f"fold_{fold}"

		print(f"\n  提取训练集特征...")
		X_train = extract_features(X_train_raw, split_name=f"{cache_prefix}_train")

		print(f"\n  提取测试集特征...")
		X_test = extract_features(X_test_raw, split_name=f"{cache_prefix}_test")

		# 关键：保证 train/test 的 one-hot 特征列完全一致
		X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

		# 填充: 统计量只从训练集来 (fit on train, transform on test)
		col_means = X_train.mean(numeric_only=True)
		X_train   = X_train.fillna(col_means).fillna(0)
		X_test    = X_test.fillna(col_means).fillna(0)

		feature_names = list(X_train.columns)   # 保存特征名，SHAP 图需要用

		# 训练和预测
		if model_name == 'xgb':
			booster, y_pred = xgb_train_and_predict(X_train, y_train, X_test, fold_idx)

			# ★ 改动三：对账断言（预测出来后、任何保存动作之前）★
			# 冻结 npy 是 np.maximum(y_pred, 0) 之后保存的，故按 clip 后口径对账
			frozen = np.load(FROZEN_DIR / 'predictions_xgb' / f'pred_{pred_suffix}.npy')
			replay = np.maximum(y_pred, 0)          # 与冻结口径一致: clip 后对账
			diff = np.abs(replay - frozen)
			print(f'对账: max diff = {diff.max():.3e}, n = {len(diff)}')
			assert diff.max() == 0.0, '重放与冻结不等价 — 曲线作废，走方案二'
			assert booster.best_iteration == 7999, f'best_iteration={booster.best_iteration} != 7999'
			print('✅ 对账通过：重放与冻结逐位等价，best_iteration=7999')

			run_shap_analysis(booster, X_train, feature_names, fold_idx, model_name)
			plot_feature_importance(booster, feature_names, fold_idx, model_name)
		else:
			model = make_model(model_name)
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
			run_shap_analysis(model, X_train, feature_names, fold_idx, model_name)
			plot_feature_importance(model, feature_names, fold_idx, model_name)

		y_pred = np.maximum(y_pred, 0)  # 物理约束: 带隙非负
		np.save(pred_file, y_pred)
		print(f'\n  → 预测结果已保存: {pred_file}')

		if not TEST_MODE:
			task_local.record(fold, y_pred)
			print(f'  → 已提交 fold {fold} 给 matbench')

	# 计算分数
	if TEST_MODE:
		scores = {'说明': 'TEST_MODE=True，不是官方分数', '模型': model_name}
		print('\n⚠️  TEST_MODE=True，跳过官方评分')
	elif len(folds_to_run) < len(list(task_local.folds)):
		scores = {'说明': f'重放模式，只跑了 fold {folds_to_run}，不出官方分数', '模型': model_name}
		print('\n⚠️  部分 fold 重放，跳过官方评分（matbench 需 5 个 fold 全部记录）')
	else:
		scores = task_local.scores
		print(f'\n✅ 官方分数: {scores}')

	metrics_path = OUT_DIR / f'scores_{model_name}{"_testmode" if TEST_MODE else ""}.txt'
	with metrics_path.open('w', encoding='utf-8') as f:
		f.write(f'任务: {TASK_NAME}\n模型: {model_name}\n\n分数:\n{str(scores)}\n')
	print(f'✅ 分数已保存: {metrics_path}')
	return scores

print('✅ 主函数定义完成')

# %%
# ==================== Cell 9：运行！ ====================
# 上面所有 cell 都是「准备工具」，这里才是「开始干活」
#
# 跑之前确认：
#   ✅ Cell 1~8 都已经跑过了
#   ✅ MODELS_TO_RUN 是你想跑的模型
#   ✅ TEST_MODE=True（先测试）或 False（正式跑）

all_scores = {}
for model_name in MODELS_TO_RUN:
	print(f'\n{"="*60}')
	print(f'=== 开始跑模型: {model_name} ===')
	print(f'{"="*60}')
	all_scores[model_name] = run_one_model(model_name)

print(f'\n{"="*60}')
print('✅ 全部完成！分数汇总:')
for k, v in all_scores.items():
	print(f'  {k}: {v}')
print('='*60)

# %%
# ==================== OOF: RF vs XGB 对比可视化 ====================

RUN_OOF_COMPARISON = False   # ★ 重放运行只跑 fold 0 训练本身；OOF 对比与本次目的无关，关掉 ★
OOF_MODELS_TO_RUN = ["rf", "xgb"]

def run_oof_comparison_on_fold0():
	"""
	在 matbench fold_0 的训练集上做标准 OOF。
	目的：
		1. 对比 RF 和 XGB 在训练集上的泛化表现
		2. 画 true vs predicted 图
		3. 导出误差最大的样本

	注意：
		这里不是 matbench 官方 test 分数。
		这里只是在 fold_0 的 train_and_val_data 内部再做 KFold。
	"""
	print("\n" + "="*60)
	print("开始 OOF 对比: RF vs XGB")
	print("="*60)

	# 重新加载 task，避免和前面 record 状态混在一起
	mb_oof = MatbenchBenchmark(autoload=False)
	task_oof = next(t for t in mb_oof.tasks if t.dataset_name == TASK_NAME)
	task_oof.load()

	# 使用 matbench fold_0 的训练集
	df = task_oof.get_train_and_val_data(0, as_type="df")
	df = df.rename(columns=lambda x: x.strip())

	if TEST_MODE:
		print(f"⚠️ TEST_MODE=True：OOF 只取前 {TEST_N} 条，图只能测试流程，不代表真实性能")
		df = df.iloc[:TEST_N].copy()

	mbids = np.array([str(x) for x in df.index], dtype=object)
	y = df[target_col].to_numpy(dtype=float)
	X_raw = df["structure"].tolist()

	print(f"OOF 样本数: {len(y)}")

	# 关键改动：
	# 不再读取旧的 FEATURES_CACHE，而是调用 extract_features。
	# extract_features 内部会按 featurizer 分别缓存。
	X_df = extract_features(X_raw, split_name="fold_0_train")
	# X_df 保留 NaN，填充在每个 OOF fold 内用训练子集统计量执行
	X = X_df

	if len(X) != len(y):
		raise ValueError(f"特征和标签数量不匹配: X={len(X)}, y={len(y)}")

	rf_params = {
		"n_estimators": 300,
		"random_state": SEED,
		"n_jobs": N_JOBS,
	}

	xgb_params = {
		"objective": "reg:squarederror",
		"eval_metric": "mae",
		"tree_method": "hist",
		"seed": SEED,
		"eta": 0.03,
		"max_depth": 6,
		"min_child_weight": 5,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"lambda": 1.0,
		"alpha": 1e-3,
	}

	kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
	oof_dict = {
		model: np.full(len(y), np.nan)
		for model in OOF_MODELS_TO_RUN
	}

	for inner_fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
		print(f"\nOOF inner fold {inner_fold+1}/{N_SPLITS}")

		X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
		X_va, y_va = X.iloc[va_idx], y[va_idx]

		# 填充: 只用本 OOF fold 训练子集的均值
		col_means = X_tr.mean(numeric_only=True)
		X_tr = X_tr.fillna(col_means).fillna(0).to_numpy()
		X_va = X_va.fillna(col_means).fillna(0).to_numpy()

		# ---------- RF ----------
		if "rf" in OOF_MODELS_TO_RUN:
			rf = RandomForestRegressor(**rf_params)
			rf.fit(X_tr, y_tr)
			oof_dict["rf"][va_idx] = np.maximum(rf.predict(X_va), 0)  # clip

			mae_rf = mean_absolute_error(y_va, oof_dict["rf"][va_idx])
			print(f"[RF]  Fold MAE: {mae_rf:.4f} eV")

		# ---------- XGB ----------
		if "xgb" in OOF_MODELS_TO_RUN:
			# XGBoost 内部再切一小块做 early stopping
			inner_tr_idx, inner_va_idx = train_test_split(
				np.arange(len(X_tr)),
				test_size=0.1,
				random_state=SEED,
				shuffle=True,
			)

			dtr = xgb.DMatrix(X_tr[inner_tr_idx], label=y_tr[inner_tr_idx])
			dval = xgb.DMatrix(X_tr[inner_va_idx], label=y_tr[inner_va_idx])
			douter = xgb.DMatrix(X_va)

			booster = xgb.train(
				params=xgb_params,
				dtrain=dtr,
				num_boost_round=8000 if not TEST_MODE else 500,
				evals=[(dval, "valid")],
				early_stopping_rounds=200 if not TEST_MODE else 50,
				verbose_eval=False,
			)

			best_it = booster.best_iteration

			try:
				pred = booster.predict(douter, iteration_range=(0, best_it + 1))
			except TypeError:
				pred = booster.predict(douter, ntree_limit=best_it + 1)

			oof_dict["xgb"][va_idx] = np.maximum(pred, 0)  # clip

			mae_xgb = mean_absolute_error(y_va, oof_dict["xgb"][va_idx])
			print(f"[XGB] Fold MAE: {mae_xgb:.4f} eV, best_it={best_it}")

	# 保存 OOF 结果
	for model in OOF_MODELS_TO_RUN:
		if np.isnan(oof_dict[model]).any():
			raise ValueError(f"{model} 的 OOF 预测里还有 NaN，说明有些样本没有被预测到")

		np.savez(
			OUT_DIR / f"oof_std_{model}{'_testmode' if TEST_MODE else ''}.npz",
			mbid=mbids,
			y_true=y,
			y_pred=oof_dict[model],
		)
		print(f"✅ {model} OOF 预测已保存")

	# 画图函数
	def plot_oof(model_name, y_true, y_pred):
		mae = mean_absolute_error(y_true, y_pred)
		rmse = np.sqrt(mean_squared_error(y_true, y_pred))
		r2 = r2_score(y_true, y_pred)

		plt.figure(figsize=(6, 6))
		plt.scatter(y_true, y_pred, alpha=0.35, s=10)

		mn = min(y_true.min(), y_pred.min())
		mx = max(y_true.max(), y_pred.max())
		plt.plot([mn, mx], [mn, mx], linestyle="--", label="Perfect Prediction")

		plt.xlabel("True Gap (eV)")
		plt.ylabel("Predicted Gap (eV)")
		plt.title(f"OOF Prediction vs True ({model_name})")

		plt.text(
			0.05,
			0.95,
			f"MAE={mae:.4f} eV\nRMSE={rmse:.4f} eV\nR²={r2:.4f}",
			transform=plt.gca().transAxes,
			va="top",
			bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
		)

		plt.tight_layout()
		fig_path = OUT_DIR / f"oof_pred_vs_true_{model_name}{'_testmode' if TEST_MODE else ''}.png"
		plt.savefig(fig_path, dpi=200)
		plt.close()
		print(f"✅ OOF 对比图保存: {fig_path}")

	# 导出 Top error
	def export_topk(model_name, mbids, y_true, y_pred, topk=TOPK):
		abs_err = np.abs(y_true - y_pred)
		idx = np.argsort(abs_err)[::-1][:topk]

		df_top = pd.DataFrame({
			"mp_id": mbids[idx],
			"true_gap_eV": y_true[idx],
			"pred_gap_eV": y_pred[idx],
			"abs_error_eV": abs_err[idx],
		})

		out_csv = OUT_DIR / f"abs_err_top{topk}_{model_name}{'_testmode' if TEST_MODE else ''}.csv"
		df_top.to_csv(out_csv, index=False)
		print(f"✅ Top{topk} 错误样本保存: {out_csv}")
		return df_top

	for model in OOF_MODELS_TO_RUN:
		plot_oof(model, y, oof_dict[model])
		export_topk(model, mbids, y, oof_dict[model])

	print("✅ OOF 对比完成")

# %%
if RUN_OOF_COMPARISON:
    run_oof_comparison_on_fold0()


