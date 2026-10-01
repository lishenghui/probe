#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --job-name=rq3-LandPrep
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_land_dense_prepare_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
python ruller-paper/experiments/rq3/extract_pool_spectra.py \
  --pool land --base "$BASE" --spec artifacts/rq3/results/land12_task_grid.json \
  --output artifacts/rq3/results/spectra_land12.json
python ruller-paper/experiments/rq3/dense_exact_lorare_allocation.py \
  --spectra artifacts/rq3/results/spectra_land12.json \
  --anchor artifacts/rq3/results/land12_anchor128.json \
  --task artifacts/rq3/results/land12_task_grid.json \
  --budget 1321 --fixed-b 1.9761636946920416 \
  --output artifacts/rq3/results/dense_exact_land12_b1321.json
