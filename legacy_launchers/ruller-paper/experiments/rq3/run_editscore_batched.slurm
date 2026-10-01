#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=rq3-editscore-batch
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_editscore_batch_%x_%j.out
set -euo pipefail

WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
JOB_CACHE="/tmp/${USER}/rq3_editscore_${SLURM_JOB_ID}"
VLLM_SIF=/nobackup/proj/disk/bloom/personal/shenghui/vllm.sif
mkdir -p "$JOB_CACHE/site" "$WORKSPACE/logs" "$WORKSPACE/artifacts/rq3/hf_cache_editscore"
# EditReward-Bench is about 3 GB.  Keep one shared copy so four variants do not
# independently download and decode it on four compute nodes.
export HF_HOME="$WORKSPACE/artifacts/rq3/hf_cache_editscore"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
apptainer exec "$VLLM_SIF" /usr/bin/python3.12 -m pip install --quiet --no-deps \
  --target "$JOB_CACHE/site" datasets peft pyarrow pandas multiprocess dill xxhash \
  qwen-vl-utils python-dotenv json-repair
export PYTHONPATH="$JOB_CACHE/site:$WORKSPACE/artifacts/rq3/vendor/EditScore"

LORA_ARGS=()
if [[ "$VARIANT" != "base" ]]; then
  LORA_ARGS=(
    --lora "$WORKSPACE/artifacts/rq3/adapters/editscore/$VARIANT"
    --merged-cache "$JOB_CACHE/merged"
  )
fi

apptainer exec --nv --env PYTHONPATH="$PYTHONPATH" --env HF_HOME="$HF_HOME" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_OFFLINE=1 \
  "$VLLM_SIF" /usr/bin/python3.12 -u \
  "$WORKSPACE/ruller-paper/experiments/rq3/evaluate_editscore_batched.py" \
  --dataset-disk "$WORKSPACE/artifacts/rq3/datasets/editreward_bench" \
  --result-dir "$WORKSPACE/artifacts/rq3/results/editscore/$VARIANT" \
  --model "$WORKSPACE/artifacts/rq3/models/qwen3-vl-4b-instruct" \
  "${LORA_ARGS[@]}" \
  --batch-size "${BATCH_SIZE:-8}" --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-32768}" ${EXTRA_ARGS:-}
