#!/bin/bash
# Cheap base/full headroom gate for the five single-GPU Wan utility adapters.
# Submit with: sbatch --array=0-9%1 run_wan_utility_pilot.slurm
# UltraWan-4K is intentionally excluded: its native protocol needs a separate
# multi-GPU generation backend rather than silently evaluating it at 480p.
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --job-name=wan-util-pilot
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_util_pilot_%A_%a.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
PY=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora/bin/python
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/wan21-t2v-1.3b
POOL=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/wan-loras/pool33.json
PROTOCOL="$W/ruller-paper/experiments/rq3/wan_utility_protocol.json"
OUTPUT="$W/artifacts/rq3/results/wan-utility"
SITE="/tmp/${USER}/wan_util_generate_${SLURM_JOB_ID}"
ADAPTERS=(panowan ultrawan1k Wan2.1-1.3b-lora-aesthetics-v1 Wan2.1-1.3b-lora-highresfix-v1 Wan2.1-1.3b-lora-speedcontrol-v1)
VARIANTS=(base full)
INDEX=${SLURM_ARRAY_TASK_ID:?submit with --array=0-9}
ADAPTER=${ADAPTERS[$((INDEX / 2))]}
VARIANT=${VARIANTS[$((INDEX % 2))]}

cd "$W"
export HF_HOME="/tmp/${USER}/hf_${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" "$OUTPUT" "$W/logs" "$SITE"
"$PY" -m pip install --quiet --target "$SITE" imageio imageio-ffmpeg
export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -u "$W/ruller-paper/experiments/rq3/wan_utility_generate.py" \
  --base "$BASE" --pool "$POOL" --protocol "$PROTOCOL" \
  --adapter "$ADAPTER" --variant "$VARIANT" --output-root "$OUTPUT" \
  --prompt-indices 0 --seeds 0 --frames 33 --steps 25
