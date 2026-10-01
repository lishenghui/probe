#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=wan21_base
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan21_base_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora

WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RUN_ROOT=/tmp/${USER}/wan21_base_${SLURM_JOB_ID}
PYTHON_DEPS=$RUN_ROOT/python
HF_CACHE=$RUN_ROOT/hf_cache
OUTPUT_DIR=$WORKSPACE/artifacts/bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA/samples
mkdir -p "$PYTHON_DEPS" "$HF_CACHE" "$OUTPUT_DIR"
test "$RUN_ROOT" != /tmp

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m pip install --target "$PYTHON_DEPS" --no-deps \
    'diffusers==0.39.0' 'transformers==4.50.0' 'peft==0.17.0' 'accelerate==1.10.0' \
    'tokenizers==0.21.4' 'huggingface-hub==0.36.2' 'numpy<2.0.0' \
    sentencepiece ftfy wcwidth imageio imageio-ffmpeg
export PYTHONPATH="$PYTHON_DEPS:$WORKSPACE/tools:${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u "$WORKSPACE/tools/generate_wan21_base.py" \
    --output "$OUTPUT_DIR/wan21_base_fox_seed0_50steps.mp4"

ls -lh "$OUTPUT_DIR/wan21_base_fox_seed0_50steps.mp4" "$OUTPUT_DIR/wan21_base_fox_seed0_50steps.json"
