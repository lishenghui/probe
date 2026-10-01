#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=rq3-vlaoft-prep
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_vlaoft_prep_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
ENV=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
HF_CACHE="$W/artifacts/rq3/hf_cache_openvla_oft"
SOURCE_ROOT="$W/artifacts/rq3/models/openvla-oft-source"
VARIANT_ROOT="$W/artifacts/rq3/models/openvla-oft-variants"
mkdir -p "$HF_CACHE" "$SOURCE_ROOT" "$VARIANT_ROOT" "$W/logs"
export HF_HOME="$HF_CACHE"

if [[ ! -d "$W/artifacts/rq3/vendor/openvla-oft/.git" ]]; then
  git clone --depth 1 https://github.com/moojink/openvla-oft.git \
    "$W/artifacts/rq3/vendor/openvla-oft"
fi
if [[ ! -d "$W/artifacts/rq3/vendor/LIBERO/.git" ]]; then
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    "$W/artifacts/rq3/vendor/LIBERO"
fi

for SUITE in spatial object goal 10; do
  REPO="moojink/openvla-7b-oft-finetuned-libero-$SUITE"
  SOURCE="$SOURCE_ROOT/$SUITE"
  "$ENV/bin/hf" download "$REPO" \
    --local-dir "$SOURCE" \
    --exclude 'model-*.safetensors' --exclude 'model.safetensors.index.json'
  "$ENV/bin/python" "$W/ruller-paper/experiments/rq3/prepare_openvla_oft_variants.py" \
    --source "$SOURCE" --output-root "$VARIANT_ROOT/$SUITE"
done
