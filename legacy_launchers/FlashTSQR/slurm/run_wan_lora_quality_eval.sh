#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --job-name=wan_lora_quality
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_lora_quality_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
NV=$W/nvcc_env
RUNROOT=/tmp/${USER}/wan_lora_eval_${SLURM_JOB_ID}
CODE=$RUNROOT/LightX2V
MODEL=${WAN_MODEL_PATH:-$RUNROOT/Wan2.2-T2V-A14B}
PYDEPS=$RUNROOT/python
OUTPUT=$W/artifacts/lightx2v/Wan2.2-Distill-Loras/quality_eval
mkdir -p "$RUNROOT" "$PYDEPS" "$OUTPUT"
test "$RUNROOT" != /tmp
df -h /tmp

git clone --depth 1 https://github.com/ModelTC/LightX2V.git "$CODE"
# The upstream pipeline eagerly imports every supported model family.  This
# Wan-only evaluation should not require unrelated audio/server dependencies.
sed -i '/^from lightx2v\.models\.runners\./{/wan_distill_runner/!s/^/# Wan-only eval: /;}' \
    "$CODE/lightx2v/pipeline.py"
python -m pip install --target "$PYDEPS" --no-deps \
    loguru omegaconf antlr4-python3-runtime opencv-python-headless imageio imageio-ffmpeg \
    einops ftfy wcwidth av gguf qtorch prometheus-client pyzmq
python -m pip install --target "$PYDEPS" pydantic
export PYTHONPATH="$PYDEPS:$CODE:${PYTHONPATH:-}"
# Catch ordinary Python dependency problems before spending time on CUDA compilation.
python - <<'PY'
import lightx2v
print("LightX2V Python import preflight OK")
PY
CUDA_BUILD_ROOT="$RUNROOT/cuda"
mkdir -p "$CUDA_BUILD_ROOT"
ln -s "$NV/bin" "$CUDA_BUILD_ROOT/bin"
ln -s "$NV/targets/sbsa-linux/include" "$CUDA_BUILD_ROOT/include"
ln -s "$NV/targets/sbsa-linux/lib" "$CUDA_BUILD_ROOT/lib64"
export CUDA_HOME="$CUDA_BUILD_ROOT" PATH="$CUDA_BUILD_ROOT/bin:$PATH" MAX_JOBS=8
export CPATH="$NV/targets/sbsa-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NV/targets/sbsa-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NV/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"
# Wan2.2 inference only needs fixed-length BF16 forward at head_dim=128 on Hopper.
# Avoid building hundreds of unused training, SM80, FP8 and KV-cache kernels.
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE

git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$RUNROOT/flash-attention"
python -m pip install --target "$PYDEPS" --no-build-isolation --no-deps "$RUNROOT/flash-attention/hopper"

python - <<'PY'
import lightx2v, flash_attn_interface, torch
print("LightX2V", lightx2v.__version__)
print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0))
PY

echo "Downloading official Wan2.2 T2V base model to node-local storage"
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir "$MODEL" --max-workers 8
df -h /tmp

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROFILING_DEBUG_LEVEL=2
python -u "$W/tools/evaluate_wan_lora_truncation.py" \
    --model-path "$MODEL" \
    --config "$W/configs/wan22_t2v_lora_eval.json" \
    --adapter-root "$W/artifacts/lightx2v/Wan2.2-Distill-Loras" \
    --output-dir "$OUTPUT"
