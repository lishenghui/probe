#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --job-name=lf-cutlass-gate
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_cutlass_gate_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export PYTHONPATH="$W"
export CUDA_HOME=$W/nvcc_env
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="9.0a"
export TORCH_EXTENSIONS_DIR=$W/.torch_ext
export MAX_JOBS=16
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd "$W"
nvidia-smi --query-gpu=name --format=csv,noheader
nvcc --version | tail -2
python -u loraforge_kernels/bench_cutlass_gate.py
