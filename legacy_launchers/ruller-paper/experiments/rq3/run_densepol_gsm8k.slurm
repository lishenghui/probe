#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=dpol-gsm8k
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/densepol_gsm8k_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME"
export STAGED_V01=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
# the taus are the per-adapter operating points the dense fleet allocator selects
# at budgets 755 and 546 for gamma in {0, 0.52, 1}; evaluating them directly
# removes the snap-to-grid step the coarse sweep needed
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$STAGED_V01" \
  --work /tmp/${USER}/dpol_gsm8k_${SLURM_JOB_ID} \
  --thresholds 0.300000 0.457228 0.546173 0.739469 0.782541 0.786743 --ranks --no-margin-probe --tasks gsm8k \
  --examples 200 --batch 16 --margin-batch 4 \
  --output artifacts/rq3/results/densepol_gsm8k.json
