# Environments

The project deliberately uses two environments. The Level-1/Level-2 tree
models and downstream analyses ran in a local Conda environment, while the
Level-3 ALIGNN training and angle-mask control ran in Google Colab because of
PyTorch/DGL/ALIGNN dependency constraints. A single combined environment is
therefore neither required nor recommended.

## Tree models and downstream analysis

`tree_analysis.yml` is the portable, direct-dependency specification for the
MatBench/Matminer, random-forest, XGBoost, SHAP, and plotting workflows.

```bash
conda env create -f environments/tree_analysis.yml
conda activate mp-gap-tree-analysis
```

`tree_analysis_macos_arm64_lock.yml` is the fuller environment capture made on
17 August 2026. It is retained as an archival lock for the original platform,
not as a cross-platform specification. The recorded platform is in
`tree_analysis_platform.txt`:

- Darwin 25.5.0, arm64
- Python 3.9.23
- Conda 26.1.0

The raw `pip freeze` capture is intentionally not distributed because packages
installed by Conda were represented by non-portable `file://` build-worker
paths. The Conda export records the same package versions without those paths.

## ALIGNN training in Google Colab

Run `install_alignn_colab.sh` in a fresh GPU-enabled Colab runtime before the
final ALIGNN scripts:

```bash
bash environments/install_alignn_colab.sh
```

The archived training scripts pin PyTorch 2.4.0 (CUDA 12.4 wheel), TorchData
0.8.0, and ALIGNN 2026.5.20. They request `dgl` without a package-version pin
from DGL's PyTorch-2.4/CUDA-12.4 wheel index. The installation script reproduces
that recorded selection route and prints the resolved DGL version; it does not
turn the historical DGL installation into an exact version lock.

The complete historical Colab `pip freeze` and exact resolved DGL version were
not archived, so DGL and some transitive packages (`jarvis-tools`, `pydantic`,
`pydantic-settings`, `pymatgen`, and `lmdb`) cannot be claimed as an exact lock.
This limitation is documented rather than reconstructed retrospectively. The
project retains the frozen five-fold predictions; when those artifacts are
distributed with the code release or a versioned data archive, every reported
downstream audit and figure can be reproduced without retraining ALIGNN.

## Scope

- Use the tree/analysis environment for baseline notebooks, replay scripts,
  SHAP aggregation, error analysis, and figure generation.
- Use the Colab environment only for full ALIGNN training and the angle-mask
  control.
- Prediction submission/evaluation through the MatBench API belongs to the
  tree/analysis environment.
