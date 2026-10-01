#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=rq3-vla-prep
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_vla_prep_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora/bin/python \
  "$W/ruller-paper/experiments/rq3/make_lora_corrections.py" \
  --weights "$W/artifacts/rq3/models/openvla-libero130-base/lora_adapter/adapter_model.safetensors" \
  --config "$W/artifacts/rq3/models/openvla-libero130-base/lora_adapter/adapter_config.json" \
  --output-root "$W/artifacts/rq3/adapters/openvla130-corrections"
