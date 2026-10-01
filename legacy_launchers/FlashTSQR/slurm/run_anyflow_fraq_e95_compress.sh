#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=anyflow_fraq_e95
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/anyflow_fraq_e95_comp_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RUN_ROOT=/tmp/${USER}/anyflow_fraq_e95_${SLURM_JOB_ID}
SOURCE_DIR=$RUN_ROOT/source
OUTPUT_DIR=$WORKSPACE/artifacts/bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA/fraq
mkdir -p "$SOURCE_DIR" "$OUTPUT_DIR"
hf download bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA \
    anyflow-wan2.1-t2v-1.3b_all-linear_rank256_anyflow-sidecar.safetensors \
    --local-dir "$SOURCE_DIR"
python -u "$WORKSPACE/tools/flashmerge_energy_truncate_lora.py" \
    "$SOURCE_DIR/anyflow-wan2.1-t2v-1.3b_all-linear_rank256_anyflow-sidecar.safetensors" \
    --output-dir "$OUTPUT_DIR" --energy-thresholds 0.95
