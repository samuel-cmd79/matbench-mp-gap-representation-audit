#!/usr/bin/env bash
# Historical Google Colab installation recipe used by the final ALIGNN scripts.
# Run in a fresh GPU-enabled Colab runtime. The core versions below are taken
# directly from the archived training scripts; the original full Colab
# transitive dependency lock was not preserved.

set -euo pipefail

python -m pip install --quiet 'setuptools<82'
python -m pip install --quiet torch==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install --quiet torchdata==0.8.0
python -m pip install --quiet dgl \
  -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
python -m pip install --quiet \
  jarvis-tools pydantic pydantic-settings pymatgen lmdb
python -m pip install --quiet alignn==2026.5.20 --no-deps

python - <<'PY'
import platform

import alignn
import dgl
import torch

print(f"Python : {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"DGL    : {dgl.__version__}")
print(f"ALIGNN : {alignn.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
PY

