# %% ============================================================
# v4: ALIGNN on matbench_mp_gap —— 修正版
# Source-informed by ALIGNN training and data-loading interfaces.
# See the repository-root THIRD_PARTY_NOTICES.md for NIST terms and attribution.
#
# 针对 alignn==2026.5.20 源码逐行核对后重写。修正内容：
#   1. [致命] v3 用 model.name="alignn"，该版本 train_dgl 只有
#      `if "alignn_" in name` 分支有训练循环 → v3 一个 epoch 都没训。
#      现改用 "alignn_atomwise" + 全零辅助权重 = 纯图级回归（等价经典 ALIGNN）。
#   2. [致命] 删除吞掉 UnboundLocalError、把随机权重存成 best_model.pt 的
#      fallback。任何异常直接 raise。
#   3. cutoff/max_neighbors 等图参数真正传入 loaders（v3 没传，全是默认值）。
#   4. 训练后强制验证门(gate)：BatchNorm 统计量 + 内部测试集 Pearson r / MAE
#      + history_val 无 NaN。不通过则中止，绝不写完成标记、绝不进入下一个 fold。
#   5. 推理复用 alignn 自己的 dataset/collate 管线，与训练 100% 同源，
#      批量前向（比 v3 逐结构快一个量级）。输出不做任何"反归一化"。
# ================================================================

# %% Cell 1 —— 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# %% Cell 2 —— 安装（版本全部钉死；GPU runtime）
# !pip install 'setuptools<82' --quiet
# !pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 --quiet
# !pip install 'torchdata==0.8.0' --quiet
# !pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html --quiet
# !pip install jarvis-tools pydantic pydantic-settings pymatgen lmdb --quiet
# !pip install alignn==2026.5.20 --no-deps --quiet
# import alignn, torch
# assert alignn.__version__ == "2026.5.20", alignn.__version__
# assert torch.cuda.is_available(), "请切换 GPU runtime"

# %% Cell 3 —— 配置与工具函数
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CFG = {
    # ---- 路径：按你的 Drive 布局改 ----
    'GNN_EXPORT_DIR': Path('/content/drive/MyDrive/matbench_alignn/gnn_export'),
    'RESULTS_DIR':    Path('/content/drive/MyDrive/matbench_alignn/results_v4'),
    'WORK_DIR':       Path('/content/alignn_work'),   # 本地盘，训练在这里跑，完了再拷 Drive

    # ---- 训练超参 ----
    'EPOCHS': 100,          # T4 大约 3-6 min/epoch；预算紧可降到 50
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 1e-3,
    'CUTOFF': 8.0,          # 论文值；这次是真的传进去了
    'MAX_NEIGHBORS': 12,
    'TRAIN_RATIO': 0.90,    # 内部切分（在 matbench 的 train+val 之内）
    'VAL_RATIO': 0.05,
    'TEST_RATIO': 0.05,     # 内部测试集，用作训练健康度的验证门
    'RANDOM_SEED': 123,

    # ---- 验证门阈值（不达标 = 训练有问题，中止）----
    'GATE_MIN_PEARSON_R': 0.85,
    'GATE_MAX_MAE': 0.60,   # eV；健康的 50-100 epoch 应在 0.3-0.45

    'FOLDS': [0, 1, 2, 3, 4],
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_fold_pkls(fold):
    """读取 v3 导出的 pkl（结构与标签由 matbench API 生成，无需改动）。"""
    d = CFG['GNN_EXPORT_DIR'] / f'fold_{fold}'
    with open(d / 'train_inputs.pkl', 'rb') as f:
        train_inputs = pickle.load(f)
    with open(d / 'train_outputs.pkl', 'rb') as f:
        train_outputs = pickle.load(f)
    with open(d / 'test_inputs.pkl', 'rb') as f:
        test_inputs = pickle.load(f)
    with open(d / 'test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    return train_inputs, train_outputs, test_inputs, test_ids


def to_dataset_array(structures, targets, prefix):
    """pymatgen Structure 列表 -> alignn dataset_array 格式。"""
    from jarvis.core.atoms import pmg_to_atoms
    arr = []
    for i, (s, y) in enumerate(zip(structures, targets)):
        arr.append({
            'jid': f'{prefix}-{i}',
            'atoms': pmg_to_atoms(s).to_dict(),
            'target': float(y),
        })
    return arr


def make_config(output_dir):
    """训练配置。关键：name='alignn_atomwise'，辅助任务全零 => 纯图级回归。"""
    from alignn.config import TrainingConfig
    from alignn.models.alignn_atomwise import ALIGNNAtomWiseConfig

    model = ALIGNNAtomWiseConfig(
        name='alignn_atomwise',
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92,       # cgcnn 特征
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=256,
        output_features=1,
        link='identity',
        # ---- 关掉一切原子级/力/应力分支 ----
        graphwise_weight=1.0,
        gradwise_weight=0.0,          # =0 时模型自动置 calculate_gradient=False
        stresswise_weight=0.0,
        atomwise_weight=0.0,
        atomwise_output_features=0,
        calculate_gradient=False,
        additional_output_features=0,
        additional_output_weight=0,
    )
    return TrainingConfig(
        version='v4',
        dataset='user_data',
        target='target',
        id_tag='jid',
        random_seed=CFG['RANDOM_SEED'],
        atom_features='cgcnn',
        neighbor_strategy='k-nearest',
        use_canonize=True,
        cutoff=CFG['CUTOFF'],
        max_neighbors=CFG['MAX_NEIGHBORS'],
        compute_line_graph=True,
        train_ratio=CFG['TRAIN_RATIO'],
        val_ratio=CFG['VAL_RATIO'],
        test_ratio=CFG['TEST_RATIO'],
        epochs=CFG['EPOCHS'],
        batch_size=CFG['BATCH_SIZE'],
        learning_rate=CFG['LEARNING_RATE'],
        weight_decay=1e-5,
        optimizer='adamw',
        scheduler='onecycle',
        write_predictions=True,       # 训完自动写内部 train/test 预测 csv
        write_checkpoint=True,
        save_dataloader=False,
        use_lmdb=True,                # 非 LMDB 路径在 2026.5.20 有 kwarg bug，必须走 LMDB
        num_workers=4,                # =0 时 A100 会被数据管道饿死（v4 首跑 6min/epoch 的元凶）
        pin_memory=True,
        dtype='float32',
        output_dir=str(output_dir),
        model=model,
    )


def build_loaders(dataset_array, config):
    """手动建 loaders 后传给 train_dgl（train_alignn.py 官方入口同款做法）。"""
    from alignn.data import get_train_val_loaders
    return get_train_val_loaders(
        dataset=config.dataset,
        dataset_array=dataset_array,
        target=config.target,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=False,            # cgcnn 特征不标准化（train.py 同款逻辑）
        line_graph=True,
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=False,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_features=1,
        classification_threshold=None,
        target_multiplication_factor=None,
        standard_scalar_and_pca=False,
        keep_data_order=False,
        output_dir=config.output_dir,
        use_lmdb=config.use_lmdb,
        dtype=config.dtype,
    )


# %% Cell 4 —— 验证门与推理
import threading


def start_backup_thread(src_dir, dst_dir, interval=600):
    """每 10 分钟把 checkpoint/历史备份到 Drive，防止 runtime 回收时全部丢失。"""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def loop():
        names = ['best_model.pt', 'current_model.pt', 'config.json',
                 'history_train.json', 'history_val.json']
        while not stop.wait(interval):
            for name in names:
                s = src_dir / name
                if s.exists():
                    try:
                        shutil.copy2(s, dst_dir / name)
                    except Exception:
                        pass

    threading.Thread(target=loop, daemon=True).start()
    return stop


def verify_training(output_dir):
    """训练健康度硬检查。任何一条不过 => RuntimeError，中止流程。"""
    output_dir = Path(output_dir)

    # 1. best_model.pt 必须由训练循环产生
    ckpt = output_dir / 'best_model.pt'
    if not ckpt.exists():
        raise RuntimeError('best_model.pt 不存在：val 从未改善或训练未运行')

    # 2. 若架构含 BatchNorm，其统计量不能是初始全零（全零 = 没见过任何 batch）。
    #    注意 ALIGNNAtomWise 用 LayerNorm，没有 running_mean，此检查自动跳过。
    state = torch.load(ckpt, map_location='cpu')
    bn = [v for k, v in state.items() if 'running_mean' in k]
    if bn and all(float(v.abs().max()) == 0.0 for v in bn):
        raise RuntimeError('BatchNorm running_mean 全零：模型未经过训练')

    # 3. history_val 不能有 NaN
    hist = json.load(open(output_dir / 'history_val.json'))
    val_losses = [row[0] for row in hist]
    if any(not np.isfinite(v) for v in val_losses):
        raise RuntimeError(f'val loss 出现 NaN/Inf: {val_losses[:5]} ...')
    print(f'  val loss: 首 {val_losses[0]:.4f} -> 末 {val_losses[-1]:.4f} '
          f'(最好 {min(val_losses):.4f})')

    # 4. 内部测试集：r 与 MAE 达标
    df = pd.read_csv(output_dir / 'prediction_results_test_set.csv',
                     skipinitialspace=True)
    y, p = df['target'].values, df['prediction'].values
    mae = float(np.mean(np.abs(y - p)))
    r = float(np.corrcoef(y, p)[0, 1])
    print(f'  内部测试集: MAE={mae:.4f} eV, Pearson r={r:.4f}, n={len(df)}')
    if r < CFG['GATE_MIN_PEARSON_R'] or mae > CFG['GATE_MAX_MAE']:
        raise RuntimeError(
            f'验证门未通过 (r={r:.3f} < {CFG["GATE_MIN_PEARSON_R"]} 或 '
            f'MAE={mae:.3f} > {CFG["GATE_MAX_MAE"]})，训练质量异常')
    return {'internal_test_mae': mae, 'internal_test_r': r}


def load_model(output_dir):
    from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
    mcfg = json.load(open(Path(output_dir) / 'config.json'))['model']
    net = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**mcfg))
    net.load_state_dict(torch.load(Path(output_dir) / 'best_model.pt',
                                   map_location='cpu'))
    return net.to(DEVICE).eval()


def predict(net, structures, scratch_dir, batch_size=64):
    """用与训练完全同源的 LMDB dataset/collate 管线做批量推理，保持输入顺序。"""
    from alignn.lmdb_dataset import get_torch_dataset
    from torch.utils.data import DataLoader

    scratch_dir = Path(scratch_dir); scratch_dir.mkdir(parents=True, exist_ok=True)
    arr = to_dataset_array(structures, [0.0] * len(structures), 'pred')
    data = get_torch_dataset(
        dataset=arr, id_tag='jid', target='target',
        atom_features='cgcnn', neighbor_strategy='k-nearest',
        use_canonize=True, name='user_data', line_graph=True,
        cutoff=CFG['CUTOFF'], max_neighbors=CFG['MAX_NEIGHBORS'],
        classification=False, output_dir=str(scratch_dir),
        tmp_name=str(scratch_dir / 'predict_lmdb'), dtype='float32',
        read_existing=False,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        collate_fn=data.collate_line_graph, drop_last=False)
    preds = []
    with torch.no_grad():
        for g, lg, lat, _ in loader:
            out = net([g.to(DEVICE), lg.to(DEVICE), lat.to(DEVICE)])['out']
            preds.append(out.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds)


# %% Cell 5 —— 主循环
def run_fold(fold):
    print(f'\n{"="*60}\nFOLD {fold}\n{"="*60}')
    t0 = time.time()

    work = CFG['WORK_DIR'] / f'fold_{fold}'
    out = work / 'model_output'
    drive_out = CFG['RESULTS_DIR'] / f'fold_{fold}'
    drive_out.mkdir(parents=True, exist_ok=True)

    if (drive_out / 'VERIFIED.json').exists():
        print('  已完成并通过验证，跳过')
        return

    out.mkdir(parents=True, exist_ok=True)
    train_inputs, train_outputs, test_inputs, test_ids = load_fold_pkls(fold)
    print(f'  训练 {len(train_inputs)} / 测试 {len(test_inputs)}')

    # ---- 训练（无 try/except：出错就让它当场炸）----
    from alignn.train import train_dgl
    config = make_config(out)
    loaders = build_loaders(
        to_dataset_array(train_inputs, train_outputs, f'f{fold}'), config)
    backup_stop = start_backup_thread(out, drive_out / 'checkpoint_backup')
    try:
        train_dgl(config, model=None, train_val_test_loaders=list(loaders))
    finally:
        backup_stop.set()

    # ---- 验证门：不过则 raise，绝不写完成标记 ----
    print('  验证训练质量...')
    gate = verify_training(out)

    # ---- 预测 matbench 官方测试集（不做任何缩放/反归一化）----
    print('  预测 matbench 测试集...')
    net = load_model(out)
    preds = predict(net, test_inputs, work / 'pred_scratch')
    assert len(preds) == len(test_ids)
    np.savez(drive_out / 'test_preds.npz',
             ids=np.array(test_ids), preds=preds)
    pd.DataFrame({'id': test_ids, 'prediction': preds}).to_csv(
        drive_out / 'test_preds.csv', index=False)
    print(f'  预测 mean={preds.mean():.3f} std={preds.std():.3f} '
          f'(真实 band gap std≈1.35，此处应同量级)')

    # ---- 拷贝训练产物到 Drive，最后才写 VERIFIED ----
    for name in ['best_model.pt', 'config.json', 'history_train.json',
                 'history_val.json', 'prediction_results_test_set.csv',
                 'prediction_results_train_set.csv', 'mad',
                 'ids_train_val_test.json']:
        src = out / name
        if src.exists():
            shutil.copy2(src, drive_out / name)
    gate['fold'] = fold
    gate['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(gate, open(drive_out / 'VERIFIED.json', 'w'), indent=2)
    print(f'  ✅ fold {fold} 完成并通过验证，耗时 {gate["elapsed_min"]} 分钟')


for fold in CFG['FOLDS']:
    run_fold(fold)

print('\n全部 fold 完成。用各 fold 的 test_preds.npz 走你原来的 matbench 评估即可。')
