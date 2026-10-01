#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=wan_lora_latency
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_lora_latency_%j.out
set -euo pipefail

# Denoising-step latency of Wan2.1-1.3B under the rank-256 AnyFlow sidecar and
# its FraQ-compressed variants, with the stock PEFT sidecar and with the
# LoRAForge rank-proportional one.

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RUN_ROOT=/tmp/${USER}/wan_lora_latency_${SLURM_JOB_ID}
ANYFLOW_SRC=$RUN_ROOT/AnyFlow
PYTHON_DEPS=$RUN_ROOT/python
HF_CACHE=$RUN_ROOT/hf_cache
ADAPTERS=$RUN_ROOT/adapters
OUT=$WORKSPACE/artifacts/bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA/latency/run_${SLURM_JOB_ID}
STEM=anyflow-wan2.1-t2v-1.3b_all-linear_rank256_anyflow-sidecar
mkdir -p "$PYTHON_DEPS" "$HF_CACHE" "$ADAPTERS" "$OUT"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git clone --filter=blob:none https://github.com/bghira/AnyFlow.git "$ANYFLOW_SRC"
git -C "$ANYFLOW_SRC" checkout 589b734fb3ebf5bc3eb3ce2a7d9b7958274bfc1e
python -m pip install --target "$PYTHON_DEPS" --no-deps \
    'diffusers==0.39.0' 'transformers==4.50.0' 'peft==0.17.0' 'accelerate==1.10.0' \
    'tokenizers==0.21.4' 'huggingface-hub==0.36.2' 'numpy<2.0.0' \
    'omegaconf==2.3.0' 'einops==0.8.1' sentencepiece ftfy wcwidth

export PYTHONPATH="$PYTHON_DEPS:$ANYFLOW_SRC:$WORKSPACE:${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE" HF_HUB_CACHE="$HF_CACHE/hub"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$RUN_ROOT/triton
mkdir -p "$TRITON_CACHE_DIR"
hf download bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA monkeypatch.py --local-dir "$ANYFLOW_SRC"
# load_lora_adapter needs a file, not a repo id, so fetch the uncompressed
# rank-256 sidecar next to the FraQ variants.
hf download bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA "${STEM}.safetensors" --local-dir "$ADAPTERS"

python -u "$WORKSPACE/tools/benchmark_wan_lora_latency.py" \
    --adapter-dir "$WORKSPACE/artifacts/bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA/fraq" \
    --original-adapter "$ADAPTERS/${STEM}.safetensors" \
    --energies e95 e90 e80 e50 \
    --output "$OUT/wan_lora_latency.json"
echo "RESULT_DIR=$OUT"
