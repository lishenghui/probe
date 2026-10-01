#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=rq3-videogpa-prep
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_videogpa_prep_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
PY=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora/bin/python
SRC="$W/artifacts/rq3/adapters/videogpa-i2v-original"
OUT="$W/artifacts/rq3/adapters/videogpa-i2v-compressed"

"$PY" "$W/ruller-paper/experiments/rq3/compress_adapter.py" \
  --weights "$SRC/adapter_model.safetensors" \
  --config "$SRC/adapter_config.json" \
  --output-root "$OUT" \
  --thresholds 0.99 0.95 0.90
