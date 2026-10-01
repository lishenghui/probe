#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --job-name=lf-rank-profile
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_rank_profile_%j.out
set -euo pipefail
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
SIF=/nobackup/proj/disk/bloom/personal/shenghui/vllm.sif
MODEL="$W/artifacts/rq3/models/qwen3-vl-4b-instruct"
OUT="$W/artifacts/kernel_bench/rank-profile"
SITE="/tmp/${USER}/lf_rank_profile_${SLURM_JOB_ID}/site"
export HF_HOME="$W/artifacts/rq3/hf_metadata" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/${USER}/triton_rank_profile_${SLURM_JOB_ID}"
mkdir -p "$SITE" "$OUT" "$TRITON_CACHE_DIR"
apptainer exec "$SIF" /usr/bin/python3.12 -m pip install --quiet --no-deps --target "$SITE" peft qwen-vl-utils
export PYTHONPATH="$SITE:$W"
run_one() {
  local variant=$1 adapter=$2
  apptainer exec --nv --env PYTHONPATH="$PYTHONPATH" --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE=1 --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    "$SIF" /usr/bin/python3.12 -u "$W/ruller-paper/experiments/rq4/profile_editscore_serving.py" \
    --model "$MODEL" --adapter "$adapter" --variant "$variant" --output "$OUT/$variant.json" \
    --batch-size 8 --new-tokens 16
}
run_one e100 "$W/artifacts/rq3/adapters/editscore/original"
run_one e90 "$W/artifacts/rq4/adapters/editscore-packed/e90"
