#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=copy_lora_75_100
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/copy_lora_75_100_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
DEST=$W/artifacts/hf_lora_census/remaining_75_100_adapters
CACHE=/tmp/${USER}/restore_hf_lora_${SLURM_JOB_ID}

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
python -u "$W/tools/restore_hf_lora_census_adapters.py" \
    --results "$W/artifacts/hf_lora_census/top100_rank32_64" \
    --output-dir "$DEST" \
    --cache-dir "$CACHE" \
    --first-order 75 --last-order 100

find "$DEST" -type f -name '*.safetensors' | wc -l
du -sh "$DEST"
