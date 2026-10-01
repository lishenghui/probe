#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=rq3-mbpp
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_mbpp_%x_%j.out
set -euo pipefail

: "${VARIANT:?Set VARIANT to original, e99, e95, or e90}"
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
BASE=unsloth/Qwen2.5-1.5B-Instruct
BASE_REV=b2e27ed8774d78eb2ee474cfe99d2d3b5fae11e5
ADAPTER="$W/artifacts/rq3/adapters/mbpp-qwen25/$VARIANT"
TMP_ROOT="/tmp/${USER}/rq3_mbpp_${SLURM_JOB_ID}"
MERGED="$TMP_ROOT/merged"
OUT="$W/artifacts/rq3/results/mbpp-qwen25/${RUN_LABEL:-$VARIANT}"

mkdir -p "$TMP_ROOT" "$OUT"
export HF_HOME="$TMP_ROOT/hf"
export TOKENIZERS_PARALLELISM=false
# Greedy MBPP evaluation does not benefit from FlashInfer's stochastic
# sampler.  Disable it so vLLM uses its built-in sampler and avoids a JIT
# dependency on CUDA cuRAND development headers unavailable on the nodes.
export VLLM_USE_FLASHINFER_SAMPLER=0
# FlashInfer JIT-compiles sampling kernels on GH200.  Point it at the
# workspace CUDA 12.6 toolkit; the compute-node image has only the driver.
export CUDA_HOME="$W/nvcc_env"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
test -x "$CUDA_HOME/bin/nvcc"

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
python -u "$W/ruller-paper/experiments/rq3/merge_mbpp_adapter.py" \
  --base "$BASE" --revision "$BASE_REV" \
  --adapter "$ADAPTER" --output "$MERGED"

mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/vllm_eval
cd "$OUT"
ARGS=(
  --model "$MERGED"
  --dataset mbpp
  --backend vllm
  --tp 1
  --greedy
  --parallel 16
)
if [[ "${MINI:-0}" == 1 ]]; then
  ARGS+=(--mini)
fi
evalplus.evaluate "${ARGS[@]}"
