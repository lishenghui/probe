#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=loraforge-qwen
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/loraforge_qwen_%j.out
set -euo pipefail
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
SIF=/nobackup/proj/disk/bloom/personal/shenghui/vllm.sif
MODEL="$W/artifacts/rq3/models/qwen3-vl-4b-instruct"
ADAPTER="$W/artifacts/rq4/adapters/editscore-packed/e90"
OUT="$W/artifacts/kernel_bench/qwen-integration"
SITE="/tmp/${USER}/loraforge_qwen_${SLURM_JOB_ID}/site"
export HF_HOME="$W/artifacts/rq3/hf_metadata" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/${USER}/triton_qwen_${SLURM_JOB_ID}"
mkdir -p "$SITE" "$OUT" "$TRITON_CACHE_DIR"
apptainer exec "$SIF" /usr/bin/python3.12 -m pip install --quiet --no-deps --target "$SITE" peft qwen-vl-utils
export PYTHONPATH="$SITE:$W"
for mode in peft loraforge; do
  extra=(); [[ "$mode" == loraforge ]] && extra=(--loraforge-kernel)
  apptainer exec --nv --env PYTHONPATH="$PYTHONPATH" --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE=1 --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    "$SIF" /usr/bin/python3.12 -u "$W/ruller-paper/experiments/rq4/benchmark_editscore_serving.py" \
    --model "$MODEL" --adapter "$ADAPTER" --variant "e90-$mode" --output "$OUT/$mode.json" \
    --batch-sizes 1 4 8 --warmup 2 --repeats 8 --new-tokens 64 "${extra[@]}"
done
