#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --array=0-59
#SBATCH --job-name=rq3-2LORl12
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_land12_512_oracle_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
TASKS=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2 gsm8k hellaswag viggo wikisql)
T=${TASKS[$((SLURM_ARRAY_TASK_ID % 12))]}
S=$((SLURM_ARRAY_TASK_ID / 12))
echo "task=$T rule-shard=$S/5"
# 57 curve points on the untruncated curve: a superset of the earlier 25-point
# sweep, so this serves the moderate budget (which needs k up to 328) and also
# supersedes the binding-budget oracle rather than sitting beside it.
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/l12or_${SLURM_JOB_ID} --tasks "$T" \
  --thresholds --ranks \
  --allocation artifacts/rq3/results/land12_512_oracle_alloc/k*.json \
  --examples 200 --batch 16 --no-margin-probe \
  --shard "$S" --shards 5 \
  --output "artifacts/rq3/results/land12_512_oracle_${T}_s${S}.json"
