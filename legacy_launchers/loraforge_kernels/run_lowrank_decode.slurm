#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=lf-lowrank
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_lowrank_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR="/tmp/${USER}/triton_lowrank_${SLURM_JOB_ID}"
LF_OUT="artifacts/kernel_bench/lowrank-decode/run_${SLURM_JOB_ID}"
LF_MODEL="${LF_MODEL:-/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf}"
mkdir -p "$LF_OUT" "$TRITON_CACHE_DIR"
python -u -m loraforge_kernels.test_decode_linear
python -u -m loraforge_kernels.test_static_decode
python -u -m loraforge_kernels.test_peft_integration
python -u -m loraforge_kernels.benchmark_lowrank_decode --output "$LF_OUT/linear.json"
python -u -m loraforge_kernels.benchmark_lowrank_model --model "$LF_MODEL" \
  --repeats 5 --output "$LF_OUT/eager.json" "$@"
python -u -m loraforge_kernels.benchmark_lowrank_model --model "$LF_MODEL" \
  --repeats 5 --cuda-graph --output "$LF_OUT/graph.json" "$@"
