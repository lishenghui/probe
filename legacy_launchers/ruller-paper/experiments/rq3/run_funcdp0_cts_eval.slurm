#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:25:00
#SBATCH --array=0-24
#SBATCH --job-name=rq3-fdp0ctse
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_fdp0ctse_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
I=$SLURM_ARRAY_TASK_ID
mapfile -t TASKS < <(jq -r '.allocation | keys[]' artifacts/rq3/results/funcdp0_alloc/cts_fra0_b8811.json)
TASK=${TASKS[$I]}
SPL=(); case "$TASK" in task0704|task0778|task0815|task0963|task1148|task1333) SPL=(--splits test valid --min-prompts 30);; esac
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
  --base "$BASE" --strengths artifacts/rq3/results/cts_adapter_strength.json \
  --work /tmp/${USER}/fdp0ctse_${SLURM_JOB_ID} --thresholds --ranks \
  --allocation artifacts/rq3/results/funcdp0_alloc/cts_fra0_b19202.json \
               artifacts/rq3/results/funcdp0_alloc/cts_fra0_b8811.json \
               artifacts/rq3/results/funcdp0_alloc/cts_fra0_b4924.json \
  --adapters "$TASK" "${SPL[@]}" --prompts 100 --batch 32 \
  --output "artifacts/rq3/results/funcdp0_eval_cts_${I}.json"
