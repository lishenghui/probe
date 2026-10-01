#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --job-name=rq3-specabl
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_spectral_abl_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
# Ablation: identical measured-D_JS fleet solve, incumbent Frobenius-optimal layer
# rule instead of the sensitivity-weighted one. Both budgets are scored in one job.
if [ "$SLURM_ARRAY_TASK_ID" -lt 12 ]; then
  BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
  TASKS=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2 gsm8k hellaswag viggo wikisql)
  T=${TASKS[$SLURM_ARRAY_TASK_ID]}
  python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
    --base "$BASE" --work /tmp/${USER}/sabl_${SLURM_JOB_ID} --tasks "$T" \
    --thresholds --ranks \
    --allocation artifacts/rq3/results/spectral_land12_b2289.json \
                  artifacts/rq3/results/spectral_land12_b1321.json \
    --examples 200 --batch 16 --no-margin-probe \
    --output "artifacts/rq3/results/spectral_abl_land12_${T}.json"
else
  BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf
  S=$((SLURM_ARRAY_TASK_ID - 12))
  python -u ruller-paper/experiments/rq3/lorare_task_metrics.py \
    --base "$BASE" --work /tmp/${USER}/sabl_${SLURM_JOB_ID} \
    --start 16 --prompts 34 --thresholds --ranks \
    --allocation artifacts/rq3/results/spectral_lorare_b5880.json \
                  artifacts/rq3/results/spectral_lorare_b3191.json \
    --shard "$S" --shards 12 \
    --output "artifacts/rq3/results/spectral_abl_lorare_s${S}.json"
fi
