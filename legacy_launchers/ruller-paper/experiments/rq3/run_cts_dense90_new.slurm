#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:50:00
#SBATCH --array=0-5
#SBATCH --job-name=rq3-cd90n
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_cts_dense90_new_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
# The six top-ups keep the test+valid protocol their grid was measured with.
python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
  --base "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a" --strengths artifacts/rq3/results/cts_adapter_strength.json \
  --work /tmp/${USER}/cd25n_${SLURM_JOB_ID} --thresholds --ranks \
  --allocation artifacts/rq3/results/dense_exact_cts_b19202.json \
  --adapters task0704 task0778 task0815 task0963 task1148 task1333 \
  --splits test valid --min-prompts 30 \
  --prompts 100 --batch 8 --shard "$SLURM_ARRAY_TASK_ID" --shards 6 \
  --output "artifacts/rq3/results/cts_dense90_new_s${SLURM_ARRAY_TASK_ID}.json"
