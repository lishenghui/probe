#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:30:00
#SBATCH --job-name=wandv-panowan
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_dv_panowan_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
python -u ruller-paper/experiments/rq3/wan_output_divergence.py \
  --base /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/wan21-t2v-1.3b \
  --pool /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/wan-loras/pool.json \
  --adapters panowan \
  --output artifacts/rq3/results/wan_dv_panowan.json
