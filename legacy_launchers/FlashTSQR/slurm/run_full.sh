#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=tsqr_full
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/tsqr_full_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
NV=$W/nvcc_env

# --- one-time: a separate prefix holding nvcc 12.6 (matches torch cu126) ---
if [ ! -x "$NV/bin/nvcc" ]; then
  echo "--- creating nvcc env (cuda-nvcc + cudart headers, 12.6) ---"
  mamba create -y -q -p "$NV" -c nvidia cuda-nvcc=12.6 cuda-cudart-dev=12.6 2>&1 | tail -3
fi
ls -la "$NV/bin/nvcc" 2>/dev/null || { echo "NVCC ENV FAILED"; ls $NV/bin 2>/dev/null | head; exit 1; }

mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/FlashTSQR/bench
export CUDA_HOME="$NV"
export PATH="$NV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=$W/.torch_ext
rm -rf $W/.torch_ext/tsqr_full

echo "--- toolchain ---"
command -v nvcc && nvcc --version | tail -2 || { echo "NO NVCC"; exit 1; }
ls $NV/include/cuda_runtime.h >/dev/null 2>&1 && echo "cuda_runtime.h OK" || echo "WARN: no cuda_runtime.h"
command -v ninja >/dev/null || pip install --quiet ninja
echo "--- run ---"
python -u bench_full.py
