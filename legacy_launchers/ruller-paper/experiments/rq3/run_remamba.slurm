#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=rq3-remamba
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_remamba_%x_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
JOB_CACHE="/tmp/${USER}/rq3_remamba_${SLURM_JOB_ID}"
PRED_ROOT="$WORKSPACE/artifacts/rq3/results/remamba/$VARIANT"
mkdir -p "$JOB_CACHE/hf" "$WORKSPACE/logs" "$PRED_ROOT"
export HF_HOME="$JOB_CACHE/hf"
export HF_HUB_DISABLE_XET=1
cd "$WORKSPACE/artifacts/rq3/vendor/ReMamba/LongBench/remamba/Longbench"

python -u predg.py --e \
  --datapath THUDM/LongBench --model ReMamba --out_name "$VARIANT" \
  --cfg_path remambacfg/remamba --remamba_sample_ratio 0.009 \
  --stratio 0 --compressp_ratio 0.18 --append_prompt False \
  --peft_path "$WORKSPACE/artifacts/rq3/adapters/remamba/$VARIANT" \
  --max_len 6144 \
  --model_path "$WORKSPACE/artifacts/rq3/models/mamba-2.8b-hf" \
  --tokenizer "$WORKSPACE/artifacts/rq3/models/mamba-2.8b-hf" \
  --output_dir "$PRED_ROOT"

python -u eval.py --e --name "$VARIANT" --pred_root "$PRED_ROOT"
