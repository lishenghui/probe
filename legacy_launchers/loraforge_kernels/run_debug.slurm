#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=lf-dbg
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_dbg_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export PYTHONPATH="$W"
export TRITON_CACHE_DIR="/tmp/${USER}/triton_lf_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
cd "$W"
nvidia-smi --query-gpu=name --format=csv,noheader
python -u -u loraforge_kernels/debug_copy_shrink.py
