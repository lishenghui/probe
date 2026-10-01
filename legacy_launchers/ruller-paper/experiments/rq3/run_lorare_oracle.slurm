#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --array=0-40
#SBATCH --job-name=rq3-2LORlr
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_lorare_oracle_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf
# One shard per adapter: 34 prompts and 48 new tokens make a single curve point
# cheap here, so all 57 fit in one job and base/full are paid once.
python -u ruller-paper/experiments/rq3/lorare_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/lror_${SLURM_JOB_ID} \
  --start 16 --prompts 34 --thresholds --ranks \
  --allocation artifacts/rq3/results/lorare_oracle_alloc/k*.json \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 41 \
  --output "artifacts/rq3/results/lorare_oracle_s${SLURM_ARRAY_TASK_ID}.json"
