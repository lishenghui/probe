#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --job-name=rq3-exposure
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_exposure_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
TASKS=(dbpedia glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

# Eighteen energy points plus seven fixed-rank points give a dense 25-point
# curve.  Unlike the main task sweep, keep the per-token probe: it records full
# and compressed margins, JS, and decision flips without using labels.
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/exposure_${SLURM_JOB_ID} \
  --tasks "$TASK" --examples 200 --batch 16 --margin-batch 4 \
  --thresholds .999 .995 .99 .98 .97 .96 .95 .94 .92 .90 .87 .85 .80 .75 .70 .60 .50 .30 \
  --ranks 1 2 3 4 5 6 7 \
  --token-output "artifacts/rq3/results/decision_exposure_dense_tokens" \
  --output "artifacts/rq3/results/decision_exposure_dense_${TASK}.json"
