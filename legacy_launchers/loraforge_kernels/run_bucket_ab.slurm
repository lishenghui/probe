#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --job-name=lf-bucket-ab
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_bucket_ab_%j.out
set -euo pipefail
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
SIF=/nobackup/proj/disk/bloom/personal/shenghui/vllm.sif
MODEL="$W/artifacts/rq3/models/qwen3-vl-4b-instruct"
ROUND=${BUCKET_ROUND:-ab}
OUT="$W/artifacts/kernel_bench/bucket-$ROUND"
SITE="/tmp/${USER}/lf_bucket_${SLURM_JOB_ID}/site"
export HF_HOME="$W/artifacts/rq3/hf_metadata" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/${USER}/triton_bucket_${SLURM_JOB_ID}"
mkdir -p "$SITE" "$OUT" "$TRITON_CACHE_DIR"
apptainer exec "$SIF" /usr/bin/python3.12 -m pip install --quiet --no-deps --target "$SITE" peft qwen-vl-utils
export PYTHONPATH="$SITE:$W"
run_one() {
  local variant=$1 adapter=$2
  apptainer exec --nv --env PYTHONPATH="$PYTHONPATH" --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE=1 --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    "$SIF" /usr/bin/python3.12 -u "$W/ruller-paper/experiments/rq4/benchmark_editscore_serving.py" \
    --model "$MODEL" --adapter "$adapter" --variant "$variant" --output "$OUT/$variant.json" \
    --batch-sizes 1 4 8 --warmup 2 --repeats 6 --new-tokens 64 --loraforge-kernel
}
if [[ "$ROUND" == align8 ]]; then
  run_one e90-variable "$W/artifacts/rq4/adapters/editscore-packed/e90"
  run_one e90-aligned8 "$W/artifacts/rq4/adapters/editscore-packed/e90-8"
elif [[ "$ROUND" == align8ba ]]; then
  run_one e90-aligned8 "$W/artifacts/rq4/adapters/editscore-packed/e90-8"
  run_one e90-variable "$W/artifacts/rq4/adapters/editscore-packed/e90"
elif [[ "$ROUND" == ba ]]; then
  run_one e90-bucketed "$W/artifacts/rq4/adapters/editscore-packed/e90-bucketed"
  run_one e90-variable "$W/artifacts/rq4/adapters/editscore-packed/e90"
else
  run_one e90-variable "$W/artifacts/rq4/adapters/editscore-packed/e90"
  run_one e90-bucketed "$W/artifacts/rq4/adapters/editscore-packed/e90-bucketed"
fi
