#!/bin/bash
# Merge one of 4 suites x 4 energy variants from the identical pristine base.
# Submit after prepare_openvla_oft_suites.slurm with --array=0-15.
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=rq3-vlaoft-merge
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_vlaoft_merge_%A_%a.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
ENV=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
REPO="$W/artifacts/rq3/vendor/openvla-oft"
ROOT="$W/artifacts/rq3/models/openvla-oft-variants"
SUITES=(spatial object goal 10)
VARIANTS=(e100 e99 e95 e90)
IDX=${SLURM_ARRAY_TASK_ID:?submit with --array=0-15}
SUITE=${SUITES[$((IDX / 4))]}
VARIANT=${VARIANTS[$((IDX % 4))]}
CHECKPOINT="$ROOT/$SUITE/$VARIANT"
PY_OVERLAY="$W/artifacts/rq3/python_overlays/openvla_oft"

test -f "$CHECKPOINT/lora_adapter/adapter_model.safetensors"
test -d "$REPO"
mkdir -p "$PY_OVERLAY"
# Keep the shared environment immutable and keep pip's cache off $HOME.  OpenVLA
# requires its v4.40.1 Transformers fork; the generic hfedlora environment does
# not expose AutoModelForVision2Seq.
export PIP_CACHE_DIR="$W/artifacts/rq3/pip_cache"
export PYTHONPATH="$PY_OVERLAY:$REPO${PYTHONPATH:+:$PYTHONPATH}"
if ! "$ENV/bin/python" -c \
  'import draccus, peft, timm; assert peft.__version__ == "0.11.1"; from transformers import AutoModelForVision2Seq; from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction' 2>/dev/null; then
  "$ENV/bin/python" -m pip install --quiet --upgrade --target "$PY_OVERLAY" \
    'draccus==0.8.0' 'tokenizers==0.19.1' 'peft==0.11.1' \
    'timm==0.9.10' 'einops' 'json-numpy' 'jsonlines' 'rich' \
    'sentencepiece==0.1.99' --no-deps
  "$ENV/bin/python" -m pip install --quiet --upgrade --target "$PY_OVERLAY" \
    'transformers @ git+https://github.com/moojink/transformers-openvla-oft.git' \
    --no-deps
fi
export HF_HOME="$W/artifacts/rq3/hf_cache_openvla_oft"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
"$ENV/bin/python" vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint openvla/openvla-7b \
  --lora_finetuned_checkpoint_dir "$CHECKPOINT"

test -f "$CHECKPOINT/model.safetensors.index.json"
touch "$CHECKPOINT/.merge_complete"
