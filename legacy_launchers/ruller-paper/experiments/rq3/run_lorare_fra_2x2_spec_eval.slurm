#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:30:00
#SBATCH --array=0-23
#SBATCH --job-name=rq3-fs2eval
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_fs2eval_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
R=artifacts/rq3/results/fra2x2_alloc
SHARD=$((SLURM_ARRAY_TASK_ID % 12))
METHOD=$((SLURM_ARRAY_TASK_ID / 12))
if [ "$METHOD" -eq 0 ]; then TAG=funcdp_spec_js; else TAG=funcdp_spec_flip; fi

python -u ruller-paper/experiments/rq3/lorare_task_metrics.py \
  --base /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf \
  --work /tmp/${USER}/fs2eval_${SLURM_JOB_ID} \
  --start 16 --prompts 34 --thresholds --ranks \
  --allocation "$R/fra2x2_${TAG}_b5880.json" "$R/fra2x2_${TAG}_b3191.json" "$R/fra2x2_${TAG}_b2683.json" \
  --shard "$SHARD" --shards 12 \
  --output "artifacts/rq3/results/fra2x2_eval_${TAG}_s${SHARD}.json"
