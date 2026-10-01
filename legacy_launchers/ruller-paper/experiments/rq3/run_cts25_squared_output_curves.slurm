#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-24
#SBATCH --job-name=rq3-CTS25-sq
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_cts25_squared_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" logs
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
mapfile -t TASKS < <(jq -r '.allocation | keys[]' artifacts/rq3/results/dense_exact_cts_b8811.json)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
python -u ruller-paper/experiments/rq3/land_two_level_curve.py \
  --pool cts --base "$BASE" --task "$TASK" \
  --work /tmp/${USER}/cts25sq_${SLURM_JOB_ID} \
  --examples 16 --example-start 0 --batch 4 --new 32 \
  --gamma 2 --layer-risk squared --divergence output \
  --budget-step 16 --max-budget 1536 \
  --output "artifacts/rq3/results/squared_output_cts25_${TASK}.json"
