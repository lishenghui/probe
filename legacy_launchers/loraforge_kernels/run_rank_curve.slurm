#!/bin/bash
# Same-node end-to-end rank scaling for the optimized dynamic-adapter path.
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=lf-rank-curve
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_rank_curve_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
SIF=/nobackup/proj/disk/bloom/personal/shenghui/vllm.sif
MODEL="$W/artifacts/rq3/models/qwen3-vl-4b-instruct"
OUT="$W/artifacts/kernel_bench/rank-curve-v2"
SITE="/tmp/${USER}/lf_rank_curve_${SLURM_JOB_ID}/site"
export HF_HOME="$W/artifacts/rq3/hf_metadata" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/${USER}/triton_rank_curve_${SLURM_JOB_ID}"
mkdir -p "$SITE" "$OUT" "$TRITON_CACHE_DIR"
apptainer exec "$SIF" /usr/bin/python3.12 -m pip install --quiet --no-deps --target "$SITE" peft qwen-vl-utils
export PYTHONPATH="$SITE:$W"

run_one() {
  local variant=$1 adapter=$2
  apptainer exec --nv --env PYTHONPATH="$PYTHONPATH" --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE=1 --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    "$SIF" /usr/bin/python3.12 -u "$W/ruller-paper/experiments/rq4/benchmark_editscore_serving.py" \
    --model "$MODEL" --adapter "$adapter" --variant "$variant-auto" --output "$OUT/$variant.json" \
    --batch-sizes 1 4 8 --warmup 2 --repeats 6 --new-tokens 64 --loraforge-kernel
}

# Non-monotone execution order reduces bias from temperature or filesystem warmup.
run_one e100 "$W/artifacts/rq3/adapters/editscore/original"
run_one e90  "$W/artifacts/rq4/adapters/editscore-packed/e90"
run_one e99  "$W/artifacts/rq4/adapters/editscore-packed/e99"
run_one e95  "$W/artifacts/rq4/adapters/editscore-packed/e95"
