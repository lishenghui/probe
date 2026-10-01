#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:20:00
#SBATCH --array=0-81
#SBATCH --job-name=rq3-r0lore
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_r0lore_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
METHOD=$((SLURM_ARRAY_TASK_ID / 41)); SHARD=$((SLURM_ARRAY_TASK_ID % 41))
if [ "$METHOD" -eq 0 ]; then TAG=floor1; else TAG=floor0; fi
python -u ruller-paper/experiments/rq3/lorare_task_metrics.py \
  --base /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf \
  --work /tmp/${USER}/r0lore_${SLURM_JOB_ID} \
  --start 16 --prompts 34 --thresholds --ranks \
  --allocation artifacts/rq3/results/rank0_allpools_alloc/lorare_${TAG}_b5880.json \
               artifacts/rq3/results/rank0_allpools_alloc/lorare_${TAG}_b3191.json \
               artifacts/rq3/results/rank0_allpools_alloc/lorare_${TAG}_b2683.json \
  --shard "$SHARD" --shards 41 \
  --output "artifacts/rq3/results/rank0_allpools_eval_lorare_${TAG}_${SHARD}.json"
