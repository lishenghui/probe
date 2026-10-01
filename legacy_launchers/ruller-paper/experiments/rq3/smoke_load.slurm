#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=rq3-smoke
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_smoke_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$WORKSPACE"

extra_args=()
if [[ -n "${REMAMBA_ROOT:-}" ]]; then
  extra_args+=(--remamba-root "$REMAMBA_ROOT")
fi
python -u ruller-paper/experiments/rq3/smoke_load.py \
  --modality "$MODALITY" --base "$BASE" --adapter "$ADAPTER" "${extra_args[@]}"
