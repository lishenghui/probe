#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=wan_ceiling
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_epilogue_ceiling_%j.out
set -euo pipefail

# How much of the remaining sidecar cost is the epilogue's rank-independent
# activation pass?  Run the concat path twice on one node: once for real, once
# with the expand skipped (numerically wrong, timing only).  The gap is what a
# CUTLASS-class kernel that folds the expand into the base GEMM epilogue could
# recover.  The `peft` rows are untouched by the flag and act as the control.

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RUN_ROOT=/tmp/${USER}/wan_ceiling_${SLURM_JOB_ID}
ANYFLOW_SRC=$RUN_ROOT/AnyFlow
PYTHON_DEPS=$RUN_ROOT/python
HF_CACHE=$RUN_ROOT/hf_cache
ADAPTERS=$RUN_ROOT/adapters
OUT=$WORKSPACE/artifacts/bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA/latency/ceiling_run_${SLURM_JOB_ID}
STEM=anyflow-wan2.1-t2v-1.3b_all-linear_rank256_anyflow-sidecar
mkdir -p "$PYTHON_DEPS" "$HF_CACHE" "$ADAPTERS" "$OUT"

nvidia-smi --query-gpu=name --format=csv,noheader
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
hf download bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA "${STEM}.safetensors" --local-dir "$ADAPTERS"

# Pin the variant so both runs exercise the same path and only the epilogue differs.
export LORAFORGE_VARIANT=concat

for skip in 0 1; do
  label=$([ "$skip" = 0 ] && echo real || echo no_epilogue)
  echo "=== concat epilogue: $label ==="
  LORAFORGE_SKIP_EPILOGUE=$skip python -u "$WORKSPACE/tools/benchmark_wan_lora_latency.py" \
      --adapter-dir "$WORKSPACE/artifacts/bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA/fraq" \
      --aligned-dir "$WORKSPACE/artifacts/bghira/AnyFlow-Wan2.1-T2V-1.3B-LoRA/fraq_aligned8" \
      --original-adapter "$ADAPTERS/${STEM}.safetensors" \
      --energies e90 e50 \
      --output "$OUT/wan_ceiling_${label}.json"
done
echo "RESULT_DIR=$OUT"
