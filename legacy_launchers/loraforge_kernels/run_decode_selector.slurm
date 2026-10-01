#!/bin/bash
# End-to-end validation of the small-row selector on the same Qwen workload.
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=lf-decode-select
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_decode_select_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
SIF=/nobackup/proj/disk/bloom/personal/shenghui/vllm.sif
MODEL="$W/artifacts/rq3/models/qwen3-vl-4b-instruct"
ADAPTER="$W/artifacts/rq4/adapters/editscore-packed/e90"
OUT="$W/artifacts/kernel_bench/decode-selector"
SITE="/tmp/${USER}/lf_decode_${SLURM_JOB_ID}/site"
export HF_HOME="$W/artifacts/rq3/hf_metadata" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/${USER}/triton_decode_${SLURM_JOB_ID}"
mkdir -p "$SITE" "$OUT" "$TRITON_CACHE_DIR"
apptainer exec "$SIF" /usr/bin/python3.12 -m pip install --quiet --no-deps --target "$SITE" peft qwen-vl-utils
export PYTHONPATH="$SITE:$W"

run_one() {
  local label=$1
  local decode_variant=$2
  local extra=()
  if [[ "$label" != peft ]]; then extra=(--loraforge-kernel); fi
  apptainer exec --nv --env PYTHONPATH="$PYTHONPATH" --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE=1 --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    --env LORAFORGE_DECODE_VARIANT="$decode_variant" \
    "$SIF" /usr/bin/python3.12 -u "$W/ruller-paper/experiments/rq4/benchmark_editscore_serving.py" \
    --model "$MODEL" --adapter "$ADAPTER" --variant "e90-$label" --output "$OUT/$label.json" \
    --batch-sizes 1 4 8 --warmup 2 --repeats 6 --new-tokens 64 "${extra[@]}"
}

run_one peft ""
run_one forced-row row
run_one forced-direct direct
run_one auto ""
