#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=lf-rank-stair
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_rank_stair_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
OUT="$W/artifacts/kernel_bench/rank-staircase/run_${SLURM_JOB_ID}"
export PYTHONPATH="$W"
export TRITON_CACHE_DIR="/tmp/${USER}/triton_rank_stair_${SLURM_JOB_ID}"
mkdir -p "$OUT" "$TRITON_CACHE_DIR" "$W/logs"
cd "$W"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u -m loraforge_kernels.benchmark_rank_staircase \
  --output "$OUT/rank_staircase.json" "$@"
