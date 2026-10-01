#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=rq4-level2
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq4_level2_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
SIF=/nobackup/proj/disk/bloom/personal/shenghui/vllm.sif
SITE="/tmp/${USER}/level2_bench_${SLURM_JOB_ID}/site"
export TRITON_CACHE_DIR="/tmp/${USER}/triton_l2_${SLURM_JOB_ID}"
mkdir -p "$SITE" "$TRITON_CACHE_DIR"

export PYTHONPATH="$W"

apptainer exec --nv --env PYTHONPATH="$PYTHONPATH" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  "$SIF" /usr/bin/python3.12 "$W/loraforge_kernels/benchmark_level2.py"

echo "Level 2 benchmark completed!"
