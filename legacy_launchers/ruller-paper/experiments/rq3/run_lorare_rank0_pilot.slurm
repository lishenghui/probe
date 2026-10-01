#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:30:00
#SBATCH --array=0-7
#SBATCH --job-name=rq3-rank0
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_rank0_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf
TASKS=(anli_r2_10templates multirc_10templates anli_r1_10templates story_cloze_10templates drop_10templates yelp_polarity_reviews_10templates cosmos_qa_10templates imdb_reviews_10templates)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

python -u ruller-paper/experiments/rq3/land_two_level_curve.py \
  --base "$BASE" --pool lorare --task "$TASK" \
  --work /tmp/${USER}/rank0_${SLURM_JOB_ID} \
  --examples 16 --example-start 0 --batch 4 --new 48 \
  --layer-risk spectral --min-rank 0 \
  --divergence output --record-decision-risk --budget-step 8 --max-budget 512 \
  --output "artifacts/rq3/results/rank0_spectral_lorare_${TASK}.json"
