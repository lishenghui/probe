#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-49
#SBATCH --job-name=rq3-CTS25-safe
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_cts25_safeguarded_eval_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" logs
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
mapfile -t TASKS < <(jq -r '.allocation | keys[]' artifacts/rq3/results/safeguarded_cts25_b8811.json)
BUDGETS=(8811 19202)
TASK_INDEX=$((SLURM_ARRAY_TASK_ID % 25))
BUDGET_INDEX=$((SLURM_ARRAY_TASK_ID / 25))
TASK=${TASKS[$TASK_INDEX]}
BUDGET=${BUDGETS[$BUDGET_INDEX]}
python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
  --base "$BASE" --strengths artifacts/rq3/results/cts_adapter_strength.json \
  --work /tmp/${USER}/cts25safe_${SLURM_JOB_ID} \
  --thresholds --ranks --allocation "artifacts/rq3/results/safeguarded_cts25_b${BUDGET}.json" \
  --adapters "$TASK" --splits test valid --min-prompts 30 \
  --prompts 100 --batch 8 \
  --output "artifacts/rq3/results/safeguarded_cts25_b${BUDGET}_${TASK}.json"
