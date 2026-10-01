#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=wan_merge
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_merge_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
NV=$W/nvcc_env
if [ ! -x "$NV/bin/nvcc" ]; then
  echo "--- creating nvcc env (cuda-nvcc + cudart headers, 12.6) ---"
  mamba create -y -q -p "$NV" -c nvidia cuda-nvcc=12.6 cuda-cudart-dev=12.6 2>&1 | tail -3
fi

mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
REPO=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/FlashTSQR
cd "$REPO/bench"
export CUDA_HOME="$NV"
export PATH="$NV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=$W/.torch_ext
command -v ninja >/dev/null || pip install --quiet ninja

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u bench_wan_merge.py --k 32 48 64
