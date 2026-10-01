#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --job-name=rq3-bprobe
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_batch_probe_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
# Does batch size move the measured metric? Left padding changes the padded width
# and therefore the bf16 reduction order, so this is a question about numerics,
# not about the rule being evaluated. Same adapter, same variants, three batches.
for B in 8 16 32; do
  echo "### batch $B  $(date +%s)"
  python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
    --base "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a" --strengths artifacts/rq3/results/cts_adapter_strength.json \
    --work /tmp/${USER}/bp_${SLURM_JOB_ID}_$B --ranks \
    --thresholds 0.90 0.70 --adapters task0223 --prompts 100 --batch $B \
    --output "artifacts/rq3/results/batch_probe_b${B}.json"
  echo "### done $B $(date +%s)"
done
