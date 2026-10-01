#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --job-name=probe_nvcc
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/probe_nvcc_%j.out
for M in GPU/buildenv-nvhpc/25.9-cu12.9.1-eb GPU/buildenv-gcccuda/2026.03-cu13.0; do
  echo "=========== $M ==========="
  ( module purge 2>/dev/null; module load $M 2>&1 | tail -1
    echo "nvcc: $(command -v nvcc || echo NONE)"
    command -v nvcc >/dev/null && nvcc --version | tail -2
    echo "CUDA_HOME=$CUDA_HOME CUDA_ROOT=$CUDA_ROOT CUDA_PATH=$CUDA_PATH"
    command -v nvcc >/dev/null && { R=$(dirname $(dirname $(command -v nvcc))); echo "root=$R"; ls $R | head -5; }
  )
done
echo "=========== 系统搜索 ==========="
ls -d /software/*/cuda* /software/*/*/cuda* 2>/dev/null | head -5
find /software -maxdepth 5 -name nvcc -type f 2>/dev/null | head -3
