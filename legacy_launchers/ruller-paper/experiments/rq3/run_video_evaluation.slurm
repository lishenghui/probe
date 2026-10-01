#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=rq3-eval-video
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_eval_video_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
PY=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora/bin/python
JOB_SITE="/tmp/${USER}/eval_video_${SLURM_JOB_ID}"
mkdir -p "$JOB_SITE"

"$PY" -m pip install --quiet --target "$JOB_SITE" \
  lpips scikit-image opencv-python imageio imageio-ffmpeg
export PYTHONPATH="$JOB_SITE${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="$W/artifacts/rq3/results/videogpa-i2v"

for VARIANT in original e99 e95 e90 base; do
  echo "=========================================="
  echo "Evaluating Video Quality for: $VARIANT"
  echo "=========================================="
  "$PY" "$W/ruller-paper/experiments/rq3/evaluate_video_quality.py" \
    --results-root "$OUT_DIR" \
    --variant "$VARIANT" \
    --output-json "$OUT_DIR/${VARIANT}_quality_metrics.json"
done

echo "Done evaluating video quality!"
