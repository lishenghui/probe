#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --job-name=rq3-DOa99
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_cts_anchor99_train_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export STAGED_INSTRUCT=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
python -u ruller-paper/experiments/rq3/cts_output_divergence.py \
  --base "$STAGED_INSTRUCT" --strengths artifacts/rq3/results/cts_adapter_strength.json \
  --work /tmp/${USER}/cts_anchor99_train_${SLURM_JOB_ID} \
  --thresholds 0.99 --split train --prompts 48 --dtype fp32 --chunk 2 \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 8 \
  --output "artifacts/rq3/results/cts_anchor99_train_shard${SLURM_ARRAY_TASK_ID}.json"
