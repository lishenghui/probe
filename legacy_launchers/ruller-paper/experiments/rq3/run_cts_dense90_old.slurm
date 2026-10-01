#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:50:00
#SBATCH --array=0-18
#SBATCH --job-name=rq3-cd90o
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_cts_dense90_old_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
# The nineteen original members keep the test-only protocol they were measured
# with; merging valid in here would move their base and full scores and make the
# dense point incomparable with their own threshold grid.
python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
  --base "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a" --strengths artifacts/rq3/results/cts_adapter_strength.json \
  --work /tmp/${USER}/cd25o_${SLURM_JOB_ID} --thresholds --ranks \
  --allocation artifacts/rq3/results/dense_exact_cts_b19202.json \
  --adapters task0112 task0149 task0223 task0260 task0297 task0334 task0371 task0408 task0593 task0852 task0889 task0926 task1037 task1111 task1370 task1407 task1444 task1518 task1592 \
  --prompts 100 --batch 8 --shard "$SLURM_ARRAY_TASK_ID" --shards 19 \
  --output "artifacts/rq3/results/cts_dense90_old_s${SLURM_ARRAY_TASK_ID}.json"
