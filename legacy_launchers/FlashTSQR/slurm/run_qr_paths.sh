#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=qr_paths
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/diag_qr_paths_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/FlashTSQR/bench
python -u diag_qr_paths.py
