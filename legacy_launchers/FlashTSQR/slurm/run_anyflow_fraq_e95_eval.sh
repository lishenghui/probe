#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=anyflow_e95_eval
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/anyflow_fraq_e95_eval_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RUN_ROOT=/tmp/${USER}/anyflow_e95_eval_${SLURM_JOB_ID}
ANYFLOW_SRC=$RUN_ROOT/AnyFlow
PYTHON_DEPS=$RUN_ROOT/python
HF_CACHE=$RUN_ROOT/hf_cache
FRAQ_DIR=$WORKSPACE/artifacts/bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA/fraq
OUTPUT_DIR=$FRAQ_DIR/samples
mkdir -p "$PYTHON_DEPS" "$HF_CACHE" "$OUTPUT_DIR"
git clone --filter=blob:none https://github.com/bghira/AnyFlow.git "$ANYFLOW_SRC"
git -C "$ANYFLOW_SRC" checkout 589b734fb3ebf5bc3eb3ce2a7d9b7958274bfc1e
python -m pip install --target "$PYTHON_DEPS" --no-deps \
    'diffusers==0.39.0' 'transformers==4.50.0' 'peft==0.17.0' 'accelerate==1.10.0' \
    'tokenizers==0.21.4' 'huggingface-hub==0.36.2' 'numpy<2.0.0' \
    'omegaconf==2.3.0' 'einops==0.8.1' sentencepiece ftfy wcwidth imageio imageio-ffmpeg
export PYTHONPATH="$PYTHON_DEPS:$ANYFLOW_SRC:${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE" HF_HUB_CACHE="$HF_CACHE/hub"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hf download bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA monkeypatch.py --local-dir "$ANYFLOW_SRC"
python -u "$WORKSPACE/tools/generate_anyflow_wan21_lora.py" \
    --base-model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --scheduler-model nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
    --adapter "$FRAQ_DIR/anyflow-wan2.1-t2v-1.3b_all-linear_rank256_anyflow-sidecar_fraq_e95.safetensors" \
    --output "$OUTPUT_DIR/anyflow_fraq_e95_fox_seed0.mp4"
