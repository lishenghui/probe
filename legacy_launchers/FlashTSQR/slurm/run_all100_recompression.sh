#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=lora100
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lora100_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; NV=$W/nvcc_env
: "${BENCH_METHOD:?}"; : "${BENCH_RUN_ID:?}"
OUT=$W/artifacts/hf_lora_census/unified_benchmark/$BENCH_RUN_ID/$BENCH_METHOD
mkdir -p "$OUT" "$W/logs"; cd "$W"
export CUDA_HOME="$NV" PATH="$NV/bin:$PATH" TORCH_CUDA_ARCH_LIST="9.0" TORCH_EXTENSIONS_DIR="$W/.torch_ext"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${BENCH_ISOLATE_REPOS:-0}" = 1 ]; then
  : "${BENCH_START_REPO:?}"; : "${BENCH_END_REPO:?}"
  for repo_order in $(seq "$BENCH_START_REPO" "$BENCH_END_REPO"); do
    echo "=== isolated repo order $repo_order ==="
    python -u tools/benchmark_lora_recompression_all100.py --projects 100 \
      --repo-orders "$repo_order" --max-flash-batch-size "${BENCH_MAX_FLASH_BATCH_SIZE:-128}" \
      ${BENCH_DEBUG_ARGS:-} \
      --method "$BENCH_METHOD" --output-dir "$OUT"
  done
else
  python -u tools/benchmark_lora_recompression_all100.py --projects "${BENCH_PROJECTS:-100}" ${BENCH_REPO_ARGS:-} \
    --max-flash-batch-size "${BENCH_MAX_FLASH_BATCH_SIZE:-128}" \
    ${BENCH_DEBUG_ARGS:-} \
    --method "$BENCH_METHOD" --output-dir "$OUT"
fi
