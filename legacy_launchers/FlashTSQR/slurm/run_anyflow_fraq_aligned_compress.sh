#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=anyflow_aligned
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/anyflow_fraq_aligned_%j.out
set -euo pipefail

# Recompress the AnyFlow sidecar with every retained rank rounded up to a
# multiple of 8.  Retaining more singular values can only raise the achieved
# energy; the point is that an aligned rank keeps the sidecar GEMM on cuBLAS's
# tensor-core path, which an arbitrary rank does not.

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RUN_ROOT=/tmp/${USER}/anyflow_aligned_${SLURM_JOB_ID}
SOURCE_DIR=$RUN_ROOT/source
OUTPUT_DIR=$WORKSPACE/artifacts/bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA/fraq_aligned8
STEM=anyflow-wan2.1-t2v-1.3b_all-linear_rank256_anyflow-sidecar
mkdir -p "$SOURCE_DIR" "$OUTPUT_DIR"

export CUDA_HOME=$WORKSPACE/nvcc_env
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=$RUN_ROOT/torch_ext

hf download bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA "${STEM}.safetensors" --local-dir "$SOURCE_DIR"
python -u "$WORKSPACE/tools/flashmerge_energy_truncate_lora.py" \
    "$SOURCE_DIR/${STEM}.safetensors" \
    --output-dir "$OUTPUT_DIR" --rank-multiple 8 \
    --energy-thresholds 0.95 0.90 0.80 0.50
ls -lh "$OUTPUT_DIR"
