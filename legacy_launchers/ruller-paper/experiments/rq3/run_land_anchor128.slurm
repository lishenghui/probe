#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --job-name=rq3-Anc128
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_land_anchor128_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
TASKS=(glue_qqp glue_sst2 glue_qnli hellaswag viggo gsm8k wikisql glue_cola glue_mrpc conllpp e2e_nlg dbpedia)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/land_anchor_${SLURM_JOB_ID} \
  --tasks "$TASK" --thresholds .99 .95 --examples 128 --example-start 200 \
  --batch 8 --margin-batch 2 \
  --output "artifacts/rq3/results/land_anchor128_${TASK}.json" \
  --token-output "artifacts/rq3/results/land_anchor128_${TASK}_tokens"
