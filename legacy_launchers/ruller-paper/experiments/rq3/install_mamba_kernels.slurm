#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=rq3-kernels
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_kernels_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
TARGET="$WORKSPACE/artifacts/rq3/python_packages"
CUDA_TARGET="$WORKSPACE/artifacts/rq3/cuda_env"
mkdir -p "$TARGET" "$WORKSPACE/logs"
if [[ ! -x "$CUDA_TARGET/bin/nvcc" ]]; then
  mamba create -y -p "$CUDA_TARGET" -c nvidia cuda-nvcc=12.6 cuda-cudart-dev=12.6
fi
export CUDA_HOME="$CUDA_TARGET"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/sbsa-linux/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/sbsa-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=9.0
python -m pip install --no-build-isolation --no-deps --target "$TARGET" causal-conv1d mamba-ssm
