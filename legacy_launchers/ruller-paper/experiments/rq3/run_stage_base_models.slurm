#!/bin/bash
# Stage the two base models into a shared HF cache once, so the evaluation jobs
# stop re-downloading 15 GB each. Every rq3 slurm script pointed HF_HOME at a
# job-scoped /tmp directory, which is node-local and never reused; with ~25
# evaluation jobs in this line of work that is ~375 GB of redundant transfer and
# roughly ten minutes of wall clock added to each job.
#
# No GPU: this is network and disk only, so it runs on the cpu partition.
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=rq3-stage-base
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_stage_base_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora

export HF_HOME=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1
echo "staging into $HF_HOME"
df -h /nobackup/proj/disk | tail -1

python - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo in ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"):
    p = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json", "*.model"])
    print(f"{repo} -> {p}", flush=True)
PY

echo "--- staged ---"
du -sh "$HF_HOME"
