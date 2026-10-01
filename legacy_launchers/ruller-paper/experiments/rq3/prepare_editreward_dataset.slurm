#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=rq3-editreward-data
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_editreward_data_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME="$WORKSPACE/artifacts/rq3/hf_cache_editscore"
export HF_HUB_DISABLE_XET=1
python -u - <<'PY'
from datasets import load_dataset

target = "/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/artifacts/rq3/datasets/editreward_bench"
dataset = load_dataset("EditScore/EditReward-Bench", split="train")
print(f"loaded rows={len(dataset)}", flush=True)
dataset.save_to_disk(target)
print(f"saved {target}", flush=True)
PY
