#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -J build_simvla
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/build_simvla_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/build_simvla_%j.err

# Build the CORAL/simvla env natively on an aarch64 GH200 node.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
unset PYTHONPATH
export UV_CACHE_DIR=/nobackup/proj/disk/bloom/personal/shenghui/.uv_cache
export TMPDIR=/tmp

echo "arch=$(uname -m) host=$(hostname)"
echo "uv=$(uv --version)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

ENVDIR=/nobackup/proj/disk/bloom/personal/shenghui/venvs/simvla
rm -rf "$ENVDIR"

echo "=== creating venv (python 3.11) ==="
uv venv "$ENVDIR" --python 3.11 --seed
PY="$ENVDIR/bin/python"

echo "=== installing torch 2.12.1+cu126 (aarch64) ==="
uv pip install --python "$PY" torch==2.12.1 torchvision \
  --index-url https://download.pytorch.org/whl/cu126

echo "=== installing CORAL deps ==="
uv pip install --python "$PY" \
  "transformers==4.57.6" \
  peft accelerate \
  mmengine h5py scipy einops timm safetensors pyarrow \
  opencv-python-headless json-numpy \
  pillow numpy tqdm tensorboard

echo "=== verify ==="
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda_ver", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import transformers, peft, accelerate, mmengine, h5py, cv2, scipy, einops
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
print("mmengine", mmengine.__version__)
print("cv2", cv2.__version__)
EOF

echo "=== ENV BUILD DONE: $ENVDIR ==="
du -sh "$ENVDIR"
