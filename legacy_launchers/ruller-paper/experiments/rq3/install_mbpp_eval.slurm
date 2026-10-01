#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=rq3-mbpp-setup
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_mbpp_setup_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
uv pip install \
  --python /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/vllm_eval/bin/python \
  'evalplus==0.3.1'
/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/vllm_eval/bin/python - <<'PY'
import evalplus, vllm
print("evalplus", evalplus.__version__)
print("vllm", vllm.__version__)
PY
