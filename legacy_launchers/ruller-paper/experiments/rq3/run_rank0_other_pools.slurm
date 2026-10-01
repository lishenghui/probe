#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:30:00
#SBATCH --array=0-36
#SBATCH --job-name=rq3-r0pools
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_r0pools_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
if [ "$SLURM_ARRAY_TASK_ID" -lt 12 ]; then
  mapfile -t TASKS < <(for f in artifacts/rq3/results/spectral_output_land12_n16_*.json; do jq -r .adapter "$f"; done | sort)
  TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
  python -u ruller-paper/experiments/rq3/land_two_level_curve.py \
    --pool land --base meta-llama/Llama-2-7b-hf --task "$TASK" \
    --work /tmp/${USER}/r0land_${SLURM_JOB_ID} \
    --examples 16 --example-start 0 --batch 4 \
    --layer-risk spectral --min-rank 0 --divergence output \
    --budget-step 8 --max-budget 256 \
    --output "artifacts/rq3/results/rank0_spectral_output_land12_${TASK}.json"
else
  I=$((SLURM_ARRAY_TASK_ID - 12))
  mapfile -t TASKS < <(for f in artifacts/rq3/results/spectral_output_cts25_*.json; do jq -r .adapter "$f"; done | sort)
  TASK=${TASKS[$I]}
  BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
  python -u ruller-paper/experiments/rq3/land_two_level_curve.py \
    --pool cts --base "$BASE" --task "$TASK" \
    --work /tmp/${USER}/r0cts_${SLURM_JOB_ID} \
    --examples 16 --example-start 0 --batch 4 --new 32 \
    --layer-risk spectral --min-rank 0 --divergence output \
    --budget-step 16 --max-budget 1536 \
    --output "artifacts/rq3/results/rank0_spectral_output_cts25_${TASK}.json"
fi
