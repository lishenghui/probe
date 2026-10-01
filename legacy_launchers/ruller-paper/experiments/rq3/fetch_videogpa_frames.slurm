#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=rq3-videogpa-data
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_videogpa_data_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
PY=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora/bin/python
PKG="/tmp/${USER}/videogpa_remotezip_${SLURM_JOB_ID}"
mkdir -p "$PKG"
"$PY" -m pip install --quiet --target "$PKG" remotezip
export PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$W/ruller-paper/experiments/rq3/fetch_videogpa_first_frames.py" \
  --prompts "$W/third_party/VideoGPA/dl3dv_video_captions/captions_1K.json" \
  --output "$W/artifacts/rq3/datasets/videogpa-dl3dv-first100" \
  --count 100
