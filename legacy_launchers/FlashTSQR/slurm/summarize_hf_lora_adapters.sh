#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=adapter_lora_stats
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/adapter_lora_stats_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RESULTS=$W/artifacts/hf_lora_census/top100_rank32_64
MPL=/tmp/${USER}/adapter_stats_matplotlib_${SLURM_JOB_ID}
python -u "$W/tools/summarize_hf_lora_adapters.py" "$RESULTS"
python -m pip install --quiet --target "$MPL" matplotlib
export PYTHONPATH="$MPL${PYTHONPATH:+:$PYTHONPATH}"
python -u "$W/tools/plot_hf_lora_adapters.py" "$RESULTS"
