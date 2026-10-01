#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=rq3-mbpp-compress
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_mbpp_compress_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
REPO=tokhey/qwen2.5-1.5b-mbpp-reasoning-sft
REV=4a58413a1b015fd174b4a98968b98860d7f864c8
SRC="$W/ruller/top100_hf_cache/models--tokhey--qwen2.5-1.5b-mbpp-reasoning-sft/snapshots/$REV"
OUT="$W/artifacts/rq3/adapters/mbpp-qwen25"

mkdir -p "$OUT/original"
hf download "$REPO" adapter_config.json --revision "$REV" --local-dir "$OUT/original"
ln -sfn "$SRC/adapter_model.safetensors" "$OUT/original/adapter_model.safetensors"

cd "$W"
python -u ruller-paper/experiments/rq3/compress_adapter.py \
  --weights "$SRC/adapter_model.safetensors" \
  --config "$OUT/original/adapter_config.json" \
  --output-root "$OUT"
