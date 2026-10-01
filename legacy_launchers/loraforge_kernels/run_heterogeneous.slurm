#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=lf-heterogeneous
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/lf_heterogeneous_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
LF_STAGE="/tmp/${USER}/lf_hetero_${SLURM_JOB_ID}"
LF_OUT="${LF_OUT:-$PWD/artifacts/kernel_bench/heterogeneous-fra0/run_${SLURM_JOB_ID}}"
LF_MODEL="${LF_MODEL:-/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf}"
LF_ALLOCATION="$PWD/artifacts/rq3/results/funcdp0_alloc/lorare_fra0_b3191.json"
export TRITON_CACHE_DIR="$LF_STAGE/triton"
mkdir -p "$LF_STAGE/sources" "$TRITON_CACHE_DIR" "$LF_OUT"

for task in arc_challenge story_cloze sst2; do
    hf download "Styxxxx/llama2_7b_lora-${task}" \
        adapter_model.safetensors adapter_config.json \
        --local-dir "$LF_STAGE/sources/$task"
done
python -u -m loraforge_kernels.test_compact_adapter
python -u -m loraforge_kernels.prepare_fra0_adapters \
    --allocation "$LF_ALLOCATION" --sources "$LF_STAGE/sources" \
    --output "$LF_STAGE/prepared"

# Persist the exact compact weights/proofs before timing or allocation expiry.
mkdir "$LF_OUT/compact"
cp "$LF_STAGE/prepared/preparation.json" "$LF_OUT/preparation.json"
cp "$LF_ALLOCATION" "$LF_OUT/allocation.json"
for task in arc_challenge story_cloze sst2; do
    cp -a "$LF_STAGE/prepared/$task/compact" "$LF_OUT/compact/$task"
done
python -u -m loraforge_kernels.benchmark_heterogeneous \
    --model "$LF_MODEL" --sources "$LF_STAGE/sources" \
    --prepared "$LF_STAGE/prepared" --output "$LF_OUT/measured" \
    --tasks arc_challenge story_cloze sst2 \
    --batches 1 4 --prefixes 128 512 --new-tokens 32 --repeats 5
