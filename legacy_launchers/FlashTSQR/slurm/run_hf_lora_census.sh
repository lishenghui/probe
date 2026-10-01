#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=hf_lora_census
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/hf_lora_census_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
NV=$W/nvcc_env
test -x "$NV/bin/nvcc"
export CUDA_HOME="$NV"
export PATH="$NV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR="$W/.torch_ext"
# Adapter checkpoints are temporary inputs and can easily occupy tens of GB.
# Keep them on node-local scratch; only the compact CSV/NPZ/plots are written
# back to the project filesystem.
JOB_CACHE="/tmp/${USER}/hf_lora_census_${SLURM_JOB_ID}"
mkdir -p "$JOB_CACHE/hf_home" "$JOB_CACHE/hf_cache"
export HF_HOME="$JOB_CACHE/hf_home"
cd "$W"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u tools/hf_lora_spectral_census.py \
  --limit 100 \
  --candidate-limit 1000 \
  --ranks 32 64 \
  --backend flash \
  --device cuda \
  --cache-dir "$JOB_CACHE/hf_cache" \
  --output-dir artifacts/hf_lora_census/top100_rank32_64
