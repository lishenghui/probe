#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --array=0-23
#SBATCH --job-name=rq3-LRanch
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_lorare_anchor_disjoint_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf
python -u ruller-paper/experiments/rq3/lorare_output_divergence.py \
  --base "$BASE" --strengths artifacts/rq3/results/lorare_adapter_strength.json \
  --work /tmp/${USER}/lorare_anchor_disjoint_${SLURM_JOB_ID} \
  --thresholds 0.95 --start 0 --prompts 16 --gen-batch 8 --chunk 2 \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 24 \
  --output "artifacts/rq3/results/lorare_anchor_disjoint_s${SLURM_ARRAY_TASK_ID}.json"
