#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=ad_fraq_clean
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/animatediff_fraq_clean_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RUNROOT=/tmp/${USER}/animatediff_fraq_latency_${SLURM_JOB_ID}
CODE=$RUNROOT/AnimateDiff
PYDEPS=$RUNROOT/python
BASE=$RUNROOT/sd15
FRAQ=$CODE/models/MotionLoRA/fraq
OUT=$WORKSPACE/artifacts/animatediff_fraq_stack/clean_runtime_benchmark/run_${SLURM_JOB_ID}
mkdir -p "$RUNROOT" "$PYDEPS" "$OUT"
cp -a "$WORKSPACE/third_party/AnimateDiff" "$CODE"
mkdir -p "$FRAQ" "$CODE/models/Motion_Module" "$CODE/models/MotionLoRA" "$CODE/models/DreamBooth_LoRA"
cp "$WORKSPACE/configs/animatediff_fraq_clean_benchmark.yaml" "$CODE/configs/prompts/animatediff_fraq_clean_benchmark.yaml"

python -m pip install --target "$PYDEPS" --no-deps \
  diffusers==0.11.1 transformers==4.25.1 tokenizers==0.13.3 huggingface-hub==0.14.1 \
  omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 imageio==2.27.0 \
  imageio-ffmpeg==0.4.9 einops safetensors

hf download runwayml/stable-diffusion-v1-5 \
  model_index.json tokenizer/merges.txt tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json tokenizer/vocab.json \
  text_encoder/config.json text_encoder/pytorch_model.bin vae/config.json vae/diffusion_pytorch_model.bin \
  unet/config.json unet/diffusion_pytorch_model.bin --local-dir "$BASE" --max-workers 8
hf download guoyww/animatediff mm_sd_v15_v2.ckpt --local-dir "$CODE/models/Motion_Module"
for name in ZoomIn PanRight TiltUp RollingClockwise; do
  hf download guoyww/animatediff "v2_lora_${name}.ckpt" --local-dir "$CODE/models/MotionLoRA"
done
hf download guoyww/animatediff_t2i_backups realisticVisionV60B1_v51VAE.safetensors --local-dir "$CODE/models/DreamBooth_LoRA"

export CUDA_HOME=$WORKSPACE/nvcc_env
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=$RUNROOT/torch_ext
python -u "$WORKSPACE/tools/flashmerge_weighted_stack_energy.py" \
  --checkpoint "$CODE/models/MotionLoRA/v2_lora_ZoomIn.ckpt" --weight 0.60 \
  --checkpoint "$CODE/models/MotionLoRA/v2_lora_PanRight.ckpt" --weight 0.45 \
  --checkpoint "$CODE/models/MotionLoRA/v2_lora_TiltUp.ckpt" --weight 0.35 \
  --checkpoint "$CODE/models/MotionLoRA/v2_lora_RollingClockwise.ckpt" --weight 0.25 \
  --output-dir "$FRAQ" --energy-thresholds 1.0 0.95 0.90 0.80 0.70

export PYTHONPATH="$PYDEPS:$CODE:${PYTHONPATH:-}"
cd "$CODE"
python -u -m scripts.animate --pretrained-model-path "$BASE" \
  --config configs/prompts/animatediff_fraq_clean_benchmark.yaml \
  --timing-json "$OUT/raw_timings.json" --benchmark-no-save --without-xformers
latest=$(find samples -mindepth 1 -maxdepth 1 -type d -name 'animatediff_fraq_clean_benchmark-*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
cp "$latest/config.yaml" "$OUT/config.yaml"
cp "$FRAQ/manifest.json" "$OUT/fraq_manifest.json"
printf '{"job_id":%s,"gpu":"%s"}\n' "$SLURM_JOB_ID" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > "$OUT/job.json"
