#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --job-name=lf-auto-cast
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_auto_cast_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
SIF=/nobackup/proj/disk/bloom/personal/shenghui/vllm.sif
MODEL="$W/artifacts/rq3/models/qwen3-vl-4b-instruct"
ADAPTER="$W/artifacts/rq4/adapters/editscore-packed/e90"
OUTDIR="$W/artifacts/kernel_bench/decode-selector"
SITE="/tmp/${USER}/lf_auto_cast_${SLURM_JOB_ID}/site"
export HF_HOME="$W/artifacts/rq3/hf_metadata" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/${USER}/triton_auto_cast_${SLURM_JOB_ID}"
mkdir -p "$SITE" "$TRITON_CACHE_DIR"
apptainer exec "$SIF" /usr/bin/python3.12 -m pip install --quiet --no-deps --target "$SITE" peft qwen-vl-utils
export PYTHONPATH="$SITE:$W"
run_one() {
  local label=$1 decode=$2
  apptainer exec --nv --env PYTHONPATH="$PYTHONPATH" --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE=1 --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    --env LORAFORGE_DECODE_VARIANT="$decode" \
    "$SIF" /usr/bin/python3.12 -u "$W/ruller-paper/experiments/rq4/benchmark_editscore_serving.py" \
    --model "$MODEL" --adapter "$ADAPTER" --variant "e90-$label" --output "$OUTDIR/$label.json" \
    --batch-sizes 1 4 8 --warmup 2 --repeats 6 --new-tokens 64 --loraforge-kernel
}

run_one peft-cast peft
run_one auto-cast-v2 ""
