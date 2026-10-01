#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-13
#SBATCH --job-name=rq3-intrapool
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_intraspec_pools_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
R=artifacts/rq3/results; A=$R/fra_clean_alloc
if [ "$SLURM_ARRAY_TASK_ID" -lt 2 ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then SEL="--adapters task0112 task0149 task0223 task0260 task0297 task0334 task0371 task0408 task0593 task0852 task0889 task0926 task1037 task1111 task1370 task1407 task1444 task1518 task1592"; SPL=""
  else SEL="--adapters task0704 task0778 task0815 task0963 task1148 task1333"; SPL="--splits test valid --min-prompts 30"; fi
  python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
    --base "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a" \
    --strengths $R/cts_adapter_strength.json \
    --work /tmp/${USER}/intra_${SLURM_JOB_ID} --thresholds --ranks \
    --allocation $A/cts_intra_spec_b19202.json $A/cts_intra_spec_b8811.json $A/cts_intra_spec_b4924.json \
    $SEL $SPL --prompts 100 --batch 32 \
    --output "$R/intraspec_eval_cts_${SLURM_ARRAY_TASK_ID}.json"
else
  S=$((SLURM_ARRAY_TASK_ID - 2))
  python -u ruller-paper/experiments/rq3/lorare_task_metrics.py \
    --base /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf \
    --work /tmp/${USER}/intra_${SLURM_JOB_ID} --start 16 --prompts 34 --thresholds --ranks \
    --allocation $A/lorare_intra_spec_b5880.json $A/lorare_intra_spec_b3191.json $A/lorare_intra_spec_b2683.json \
    --shard "$S" --shards 12 --output "$R/intraspec_eval_lorare_s${S}.json"
fi
