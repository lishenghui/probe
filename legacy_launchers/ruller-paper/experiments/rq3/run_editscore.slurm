#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=rq3-editscore
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_editscore_%x_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
JOB_CACHE="/tmp/${USER}/rq3_editscore_${SLURM_JOB_ID}"
mkdir -p "$JOB_CACHE/hf" "$WORKSPACE/logs" "$WORKSPACE/artifacts/rq3/results/editscore/$VARIANT"
export HF_HOME="$JOB_CACHE/hf"
export HF_HUB_DISABLE_XET=1
python -m pip install --quiet --no-deps --target "$JOB_CACHE/site" python-dotenv json-repair qwen-vl-utils
export PYTHONPATH="$JOB_CACHE/site${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORKSPACE/artifacts/rq3/vendor/EditScore"

python -u evaluation.py \
  --benchmark_dir EditScore/EditReward-Bench \
  --result_dir "$WORKSPACE/artifacts/rq3/results/editscore/$VARIANT" \
  --backbone qwen3vl \
  --model_name_or_path "$WORKSPACE/artifacts/rq3/models/qwen3-vl-4b-instruct" \
  --lora_path "$WORKSPACE/artifacts/rq3/adapters/editscore/$VARIANT" \
  --score_range 25 --max_workers 1 --max_model_len 4096 \
  --max_num_seqs 1 --max_num_batched_tokens 4096 \
  --tensor_parallel_size 1 --num_pass 1 --temperature 0.7 \
  --no_save_images

python -u calculate_statistics.py \
  --result_dir "$WORKSPACE/artifacts/rq3/results/editscore/$VARIANT/qwen3vl"
