import numpy as np
from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False, subset=["matbench_mp_gap"])
task = list(mb.tasks)[0]
task.load()

d = {}
for k, fold in enumerate(task.folds):
    _, y = task.get_test_data(fold, include_target=True)
    d[f"ids_{k}"] = np.asarray(y.index, dtype="U16")
    d[f"y_{k}"]   = np.asarray(y.values, dtype=np.float64)

np.savez("labels_by_fold.npz", **d)
print({k: v.shape for k, v in d.items()})