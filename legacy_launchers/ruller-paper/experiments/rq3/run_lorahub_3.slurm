#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=rq3-lh3
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_lorahub3_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
# Datasets must not live on the project disk: Lustre reports the project
# quota as the filesystem size, so shutil.disk_usage sees 0 bytes free and
# datasets refuses to download even though writes would succeed. Models are
# read from the shared staging path instead, which needs no space check.
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME"
export STAGED_V01=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
export STAGED_INSTRUCT=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
python -u ruller-paper/experiments/rq3/lorahub_population.py \
  --work /tmp/${USER}/lorahub_${SLURM_JOB_ID} \
  --adapters 32 --lambdas 1 2 4 8 16 --dtype fp32 --shard 3 --shards 4 \
  --output artifacts/rq3/results/lorahub_ext_shard3.json
