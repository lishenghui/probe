#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --array=0-16
#SBATCH --job-name=rq3-L17d90
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_land17_dense90_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
TASK=$(jq -r ".[$SLURM_ARRAY_TASK_ID].adapter" artifacts/rq3/results/land17_task_grid.json)
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/land17_d90_${SLURM_JOB_ID} --tasks "$TASK" \
  --thresholds --ranks --allocation artifacts/rq3/results/dense_exact_land17_b3473.json \
  --examples 200 --batch 16 --no-margin-probe \
  --output "artifacts/rq3/results/land17_dense90_${TASK}.json"
