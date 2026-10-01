#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=rq3-videogpa
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_videogpa_%x_%j.out
set -euo pipefail

: "${VARIANT:?Set VARIANT to base, original, e99, e95, or e90}"
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
R="$W/third_party/VideoGPA"
PY=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora/bin/python
JOB_SITE="/tmp/${USER}/videogpa_${SLURM_JOB_ID}/site"
BASE="$W/artifacts/rq3/models/CogVideoX-5B-I2V"
DATA="$W/artifacts/rq3/datasets/videogpa-dl3dv-first100"
PROMPTS="$R/dl3dv_video_captions/captions_1K.json"
OUT="$W/artifacts/rq3/results/videogpa-i2v/$VARIANT"

if [[ "$VARIANT" == base ]]; then
  LORA=""
  export RUN_MODE=base
elif [[ "$VARIANT" == original ]]; then
  LORA="$W/artifacts/rq3/adapters/videogpa-i2v-original"
  export RUN_MODE=dpo
else
  LORA="$W/artifacts/rq3/adapters/videogpa-i2v-compressed/$VARIANT"
  export RUN_MODE=dpo
fi
test -f "$BASE/model_index.json"
test -f "$DATA/first100_manifest.json"
if [[ -n "$LORA" ]]; then
  test -f "$LORA/adapter_model.safetensors"
fi

# Recent Transformers requires tiktoken when converting CogVideoX's T5
# tokenizer. Keep this job-local so the shared experiment environment remains
# reproducible and is not mutated by individual runs.
mkdir -p "$JOB_SITE"
"$PY" -m pip install --quiet --target "$JOB_SITE" \
  tiktoken sentencepiece protobuf imageio imageio-ffmpeg
export PYTHONPATH="$JOB_SITE${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="$W/artifacts/rq3/hf_metadata"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RUN_MODE=dpo
export RUN_BASE_MODEL="$BASE"
export RUN_LORA_PATH="$LORA"
export RUN_OUTPUT_DIR="$OUT"
export PROMPT_JSON="$PROMPTS"
export DL3DV_BASE_DIR="$DATA"
export RUN_DEVICES=0
export RUN_NUM_PROMPTS="${NUM_PROMPTS:-100}"
export RUN_SEEDS=456

cd "$R"
"$PY" replicate.py
