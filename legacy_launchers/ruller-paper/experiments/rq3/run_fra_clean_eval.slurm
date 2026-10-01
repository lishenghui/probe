#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-25
#SBATCH --job-name=rq3-fraclean
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_fra_clean_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
R=artifacts/rq3/results
A=$R/fra_clean_alloc
# Only the new safeguarded FRA union needs evaluation. E2E-Spec and FuncDP
# already have complete held-out evaluations under the same three budgets.
if [ "$SLURM_ARRAY_TASK_ID" -lt 12 ]; then
  TASKS=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2 gsm8k hellaswag viggo wikisql)
  T=${TASKS[$SLURM_ARRAY_TASK_ID]}
  python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
    --base "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da" \
    --work /tmp/${USER}/fraclean_${SLURM_JOB_ID} --tasks "$T" --thresholds --ranks \
    --allocation $A/land_fra_b2289.json $A/land_fra_b1321.json $A/land_fra_b943.json \
    --examples 200 --batch 16 --no-margin-probe \
    --output "$R/fra_clean_eval_land_${T}.json"
elif [ "$SLURM_ARRAY_TASK_ID" -lt 14 ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -eq 12 ]; then SEL="--adapters task0112 task0149 task0223 task0260 task0297 task0334 task0371 task0408 task0593 task0852 task0889 task0926 task1037 task1111 task1370 task1407 task1444 task1518 task1592"; SPL=""
  else SEL="--adapters task0704 task0778 task0815 task0963 task1148 task1333"; SPL="--splits test valid --min-prompts 30"; fi
  python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
    --base "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a" --strengths $R/cts_adapter_strength.json \
    --work /tmp/${USER}/fraclean_${SLURM_JOB_ID} --thresholds --ranks \
    --allocation $A/cts_fra_b19202.json $A/cts_fra_b8811.json $A/cts_fra_b4924.json \
    $SEL $SPL --prompts 100 --batch 32 --output "$R/fra_clean_eval_cts_${SLURM_ARRAY_TASK_ID}.json"
else
  S=$((SLURM_ARRAY_TASK_ID - 14))
  python -u ruller-paper/experiments/rq3/lorare_task_metrics.py \
    --base /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf \
    --work /tmp/${USER}/fraclean_${SLURM_JOB_ID} --start 16 --prompts 34 --thresholds --ranks \
    --allocation $A/lorare_fra_b5880.json $A/lorare_fra_b3191.json $A/lorare_fra_b2683.json \
    --shard "$S" --shards 12 --output "$R/fra_clean_eval_lorare_s${S}.json"
fi
