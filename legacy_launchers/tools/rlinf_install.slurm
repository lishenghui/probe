#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -J rlinf_install
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rlinf_install_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rlinf_install_%j.err

set -uo pipefail
RLINF=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/RLinf
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/nobackup/proj/disk/bloom/personal/shenghui/.uv_cache2
export TMPDIR=/tmp
export UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-cu128}

cd "$RLINF"
echo "arch=$(uname -m) host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

bash requirements/install.sh embodied --model openvla-oft --env libero \
    --no-root --no-flash-attn --no-apex --no-natten \
    --python 3.11

echo "=== INSTALL SCRIPT EXIT $? ==="
echo "=== venv check ==="
source .venv/bin/activate 2>/dev/null && python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())" || echo "venv activate failed"
