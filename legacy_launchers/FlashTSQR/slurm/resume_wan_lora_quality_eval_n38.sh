#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --nodelist=n38
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:30:00
#SBATCH --job-name=wan_lora_resume
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_lora_resume_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RUNROOT=/tmp/${USER}/wan_lora_eval_1185826
export PYTHONPATH="$RUNROOT/python:$RUNROOT/LightX2V:${PYTHONPATH:-}"
export CUDA_HOME="$RUNROOT/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROFILING_DEBUG_LEVEL=2
test -f "$RUNROOT/Wan2.2-T2V-A14B/config.json"
python -u "$W/tools/evaluate_wan_lora_truncation.py" \
    --model-path "$RUNROOT/Wan2.2-T2V-A14B" \
    --config "$W/configs/wan22_t2v_lora_eval.json" \
    --adapter-root "$W/artifacts/lightx2v/Wan2.2-Distill-Loras" \
    --output-dir "$W/artifacts/lightx2v/Wan2.2-Distill-Loras/quality_eval"
