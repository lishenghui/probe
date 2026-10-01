#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=truncate_wan_lora
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/truncate_wan_lora_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
NV=$W/nvcc_env
INPUT=$W/artifacts/lightx2v/Wan2.2-Distill-Loras
OUTPUT=$INPUT/fraq_truncated
export CUDA_HOME="$NV" PATH="$NV/bin:$PATH" TORCH_CUDA_ARCH_LIST="9.0" TORCH_EXTENSIONS_DIR=$W/.torch_ext
cd "$W"

for checkpoint in "$INPUT"/wan2.2_t2v_*_lora_rank64_*.safetensors; do
    python -u tools/flashmerge_truncate_lora.py "$checkpoint" \
        --output-dir "$OUTPUT" --ranks 48 32 16
done
