#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=rq3-fidelity
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_fidelity_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
PY=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora/bin/python
JOB_SITE="/tmp/${USER}/eval_fidelity_${SLURM_JOB_ID}"
mkdir -p "$JOB_SITE"

"$PY" -m pip install --quiet --target "$JOB_SITE" \
  lpips scikit-image opencv-python transformers
export PYTHONPATH="$JOB_SITE${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="$W/artifacts/rq3/results/videogpa-i2v"
CAPTIONS="$W/third_party/VideoGPA/dl3dv_video_captions/captions_1K.json"

"$PY" "$W/ruller-paper/experiments/rq3/evaluate_cross_model_fidelity.py" \
  --results-root "$OUT_DIR" \
  --captions-json "$CAPTIONS" \
  --output-json "$OUT_DIR/cross_model_fidelity_summary.json"

echo "Done evaluating cross-model fidelity and prompt alignment!"
