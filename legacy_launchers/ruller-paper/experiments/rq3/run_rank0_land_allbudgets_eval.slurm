#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --array=0-23
#SBATCH --job-name=rq3-r0lande
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_r0lande_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
TASKS=(wikisql gsm8k hellaswag glue_sst2 viggo glue_qqp glue_qnli glue_cola glue_mrpc conllpp e2e_nlg dbpedia)
I=$((SLURM_ARRAY_TASK_ID % 12)); METHOD=$((SLURM_ARRAY_TASK_ID / 12)); TASK=${TASKS[$I]}
if [ "$METHOD" -eq 0 ]; then TAG=floor1; else TAG=floor0; fi
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/r0lande_${SLURM_JOB_ID} \
  --tasks "$TASK" --thresholds --ranks \
  --allocation artifacts/rq3/results/rank0_allpools_alloc/land_${TAG}_b2289.json \
               artifacts/rq3/results/rank0_allpools_alloc/land_${TAG}_b1321.json \
               artifacts/rq3/results/rank0_allpools_alloc/land_${TAG}_b943.json \
  --examples 200 --batch 16 --no-margin-probe \
  --output "artifacts/rq3/results/rank0_allpools_eval_land_${TAG}_${I}.json"
