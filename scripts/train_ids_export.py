"""在 Mac mini 上跑，补导 train_ids.pkl"""
import pickle
from pathlib import Path
from matbench.bench import MatbenchBenchmark

GNN_EXPORT_DIR = Path("../gnn_export")

mb = MatbenchBenchmark(autoload=False, subset=["matbench_mp_gap"])
task = list(mb.tasks)[0]
task.load()

for fold in range(5):
    train_df = task.get_train_and_val_data(fold, as_type="df")
    train_ids = list(train_df.index)
    
    fold_dir = GNN_EXPORT_DIR / f"fold_{fold}"
    with open(fold_dir / "train_ids.pkl", "wb") as f:
        pickle.dump(train_ids, f)
    
    print(f"fold_{fold}: {len(train_ids)} train IDs saved")