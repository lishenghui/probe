#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=lora_method
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lora_method_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
NV=$W/nvcc_env
: "${BENCH_METHOD:?submit with --export=ALL,BENCH_METHOD=<method>}"
: "${BENCH_RUN_ID:?submit with --export=ALL,BENCH_RUN_ID=<shared run id>}"
OUT=$W/artifacts/hf_lora_census/unified_benchmark/run_${BENCH_RUN_ID}/${BENCH_METHOD}
test -x "$NV/bin/nvcc"
mkdir -p "$OUT" "$W/logs"
cd "$W"
export CUDA_HOME="$NV"
export PATH="$NV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR="$W/.torch_ext"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u tools/benchmark_lora_recompression.py \
  --projects 10 --layers-per-project "${BENCH_LAYERS_PER_PROJECT:-3}" --energy 0.95 \
  --warmup 2 --reps 5 --method "$BENCH_METHOD" ${BENCH_EXTRA_ARGS:-} --output-dir "$OUT"
echo "RESULT_DIR=$OUT"
