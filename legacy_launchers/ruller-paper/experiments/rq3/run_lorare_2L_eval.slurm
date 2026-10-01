#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:30:00
#SBATCH --array=0-11
#SBATCH --job-name=rq3-LR-g2eval
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_lorare_g2_eval_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf
# Same evaluation protocol as every other LoRARetriever number: rows 16-49 of the
# authors' eval set, disjoint from the 0-15 used to calibrate the curves.
python -u ruller-paper/experiments/rq3/lorare_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/lr2Le_${SLURM_JOB_ID} \
  --start 16 --prompts 34 --thresholds --ranks \
  --allocation artifacts/rq3/results/squared_output_lorare_g2_b3191.json \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 12 \
  --output "artifacts/rq3/results/lorare_g2_b3191_s${SLURM_ARRAY_TASK_ID}.json"
