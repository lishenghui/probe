#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctsspec
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/cts_spectra_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1
D=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache
echo "########## Lots-of-LoRAs (controlled pool) ##########"
python -u ruller-paper/experiments/rq3/extract_pool_spectra.py --pool cts \
  --base $D/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a \
  --spec artifacts/rq3/results/cts_strength_aggregations.json \
  --output artifacts/rq3/results/spectra_cts.json
