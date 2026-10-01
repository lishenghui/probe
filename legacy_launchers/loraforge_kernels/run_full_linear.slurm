#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=loraforge-full-linear
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/loraforge_full_linear_%j.out
set -euo pipefail
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
SIF=/nobackup/proj/disk/bloom/personal/shenghui/vllm.sif
export PYTHONPATH="$W" TRITON_CACHE_DIR="/tmp/${USER}/triton_full_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"; cd "$W"
apptainer exec --nv --env PYTHONPATH="$PYTHONPATH" --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
 "$SIF" /usr/bin/python3.12 -u loraforge_kernels/benchmark_full_linear.py
