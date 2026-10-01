#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --job-name=lf-vla
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_vla_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export PYTHONPATH="$W"
export TRITON_CACHE_DIR="/tmp/${USER}/triton_lf_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
cd "$W"
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__)"
python -u loraforge_kernels/bench_vla.py "$@"
