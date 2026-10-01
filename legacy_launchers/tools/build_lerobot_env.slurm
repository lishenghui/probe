#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -J build_lerobot
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/build_lerobot_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/build_lerobot_%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
unset PYTHONPATH
export UV_CACHE_DIR=/nobackup/proj/disk/bloom/personal/shenghui/.uv_cache
export TMPDIR=/tmp

echo "arch=$(uname -m) host=$(hostname)"
echo "uv=$(uv --version)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1

ENVDIR=/nobackup/proj/disk/bloom/personal/shenghui/venvs/lerobot
rm -rf "$ENVDIR"

echo "=== creating venv (python 3.12) ==="
uv venv "$ENVDIR" --python 3.12 --seed
PY="$ENVDIR/bin/python"

echo "=== torch 2.11.0+cu128 (aarch64) ==="
uv pip install --python "$PY" torch==2.11.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== lerobot[smolvla,peft] ==="
uv pip install --python "$PY" "lerobot[smolvla,peft]==0.6.1"

echo "=== verify ==="
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
import transformers, peft, lerobot, datasets, av
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("lerobot file", lerobot.__file__)
print("datasets", datasets.__version__)
import torchcodec
print("torchcodec ok")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print("SmolVLAPolicy import ok")
EOF

echo "=== ENV BUILD DONE: $ENVDIR ==="
du -sh "$ENVDIR"
