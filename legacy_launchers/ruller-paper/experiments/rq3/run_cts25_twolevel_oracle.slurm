#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:15:00
#SBATCH --array=0-99
#SBATCH --job-name=rq3-2LORcts
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_cts25_2Lor_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
ALL=(task0112 task0149 task0223 task0260 task0297 task0334 task0371 task0408 task0593 task0852 task0889 task0926 task1037 task1111 task1370 task1407 task1444 task1518 task1592 task0704 task0778 task0815 task0963 task1148 task1333)
A=${ALL[$((SLURM_ARRAY_TASK_ID % 25))]}
S=$((SLURM_ARRAY_TASK_ID / 25))              # four strided shards over 91 curve points
case "$A" in
  task0704|task0778|task0815|task0963|task1148|task1333) SPL="--splits test valid --min-prompts 30";;
  *) SPL="";;
esac
echo "adapter=$A rule-shard=$S/4"
# batch 32: a probe on task0223 timed 243/102/81 s at batch 8/16/32 and moved u by
# at most 0.008, well under this pool"s one-example resolution, so the 3x is free.
python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
  --base "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a" --strengths artifacts/rq3/results/cts_adapter_strength.json \
  --work /tmp/${USER}/cts2Lor_${SLURM_JOB_ID} --thresholds --ranks \
  --allocation artifacts/rq3/results/cts25_oracle_alloc/k*.json \
  --adapters "$A" $SPL --prompts 100 --batch 32 \
  --rule-shard "$S" --rule-shards 4 \
  --output "artifacts/rq3/results/cts25_2Lor_${A}_s${S}.json"
