#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=rq3-compress
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_compress_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora

WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$WORKSPACE"
mkdir -p logs artifacts/rq3/adapters

python -u ruller-paper/experiments/rq3/compress_adapter.py \
  --weights ruller/rank32_hf_cache/models--lblankl--ReMamba/snapshots/fa33cea292d43034c60eee576afe716cd036fc0a/adapter_model.safetensors \
  --config artifacts/rq3/adapters/remamba-original/adapter_config.json \
  --output-root artifacts/rq3/adapters/remamba

python -u ruller-paper/experiments/rq3/compress_adapter.py \
  --weights ruller/rank32_hf_cache/models--EditScore--EditScore-Qwen3-VL-4B-Instruct/snapshots/87dcfd0eba27335c036ef9616fa0e725141ba571/adapter_model.safetensors \
  --config artifacts/rq3/adapters/editscore-original/adapter_config.json \
  --output-root artifacts/rq3/adapters/editscore
