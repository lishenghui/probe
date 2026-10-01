#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:35:00
#SBATCH --array=0-15
#SBATCH --job-name=rq3-r0eval
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_r0eval_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
TASKS=(anli_r2_10templates multirc_10templates anli_r1_10templates story_cloze_10templates drop_10templates yelp_polarity_reviews_10templates cosmos_qa_10templates imdb_reviews_10templates)
METHOD=$((SLURM_ARRAY_TASK_ID / 8))
TASK_INDEX=$((SLURM_ARRAY_TASK_ID % 8))
if [ "$METHOD" -eq 0 ]; then TAG=floor1; else TAG=floor0; fi
python -u ruller-paper/experiments/rq3/lorare_task_metrics.py \
  --base /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf \
  --work /tmp/${USER}/r0eval_${SLURM_JOB_ID} \
  --start 16 --prompts 34 --thresholds --ranks \
  --tasks "${TASKS[$TASK_INDEX]}" \
  --allocation "artifacts/rq3/results/rank0_pilot_alloc/${TAG}_b576.json" \
  --output "artifacts/rq3/results/rank0_pilot_eval_${TAG}_${TASK_INDEX}.json"
