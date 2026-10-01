#!/bin/bash
# Score generated clips after the base/full pilot or compression sweep.
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=wan-util-score
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_util_score_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
PY=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora/bin/python
SITE="/tmp/${USER}/wan_util_score_${SLURM_JOB_ID}"
ROOT="$W/artifacts/rq3/results/wan-utility"
mkdir -p "$SITE" "$ROOT"
"$PY" -m pip install --quiet --target "$SITE" opencv-python-headless
export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"

cd "$W"
"$PY" "$W/ruller-paper/experiments/rq3/wan_utility_score.py" \
  --results-root "$ROOT" \
  --clip /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/clip-vit-large-p14 \
  --output "$ROOT/scores.json"
"$PY" "$W/ruller-paper/experiments/rq3/summarize_wan_utility.py" \
  "$ROOT/scores.json" --output "$ROOT/summary.json"
