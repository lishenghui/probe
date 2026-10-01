#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=plot_lora_census
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/plot_lora_census_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
MPL=/tmp/${USER}/lora_census_matplotlib_${SLURM_JOB_ID}
python -m pip install --quiet --target "$MPL" matplotlib
export PYTHONPATH="$MPL${PYTHONPATH:+:$PYTHONPATH}"
python -u "$W/tools/plot_hf_lora_census.py" \
    "$W/artifacts/hf_lora_census/top100_rank32_64"
