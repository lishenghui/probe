#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-25
#SBATCH --job-name=rq3-fsenv
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_fs_envelope_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
R=artifacts/rq3/results
# Envelope over two complementary inner proposals -- functional composition and
# weight-space spectrum -- selected per adapter and budget on the measured
# divergence alone. All three budgets are scored in one pass per adapter.
if [ "$SLURM_ARRAY_TASK_ID" -lt 12 ]; then
  TASKS=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2 gsm8k hellaswag viggo wikisql)
  T=${TASKS[$SLURM_ARRAY_TASK_ID]}
  python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
    --base "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da" --work /tmp/${USER}/fse_${SLURM_JOB_ID} --tasks "$T" \
    --thresholds --ranks \
    --allocation $R/fs_envelope_land12_b2289.json $R/fs_envelope_land12_b1321.json $R/fs_envelope_land12_b943.json \
    --examples 200 --batch 16 --no-margin-probe \
    --output "$R/fs_env_eval_land12_${T}.json"
elif [ "$SLURM_ARRAY_TASK_ID" -lt 14 ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -eq 12 ]; then SEL="--adapters task0112 task0149 task0223 task0260 task0297 task0334 task0371 task0408 task0593 task0852 task0889 task0926 task1037 task1111 task1370 task1407 task1444 task1518 task1592"; SPL=""
  else SEL="--adapters task0704 task0778 task0815 task0963 task1148 task1333"
       SPL="--splits test valid --min-prompts 30"; fi
  python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
    --base "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a" --strengths $R/cts_adapter_strength.json \
    --work /tmp/${USER}/fse_${SLURM_JOB_ID} --thresholds --ranks \
    --allocation $R/fs_envelope_cts25_b19202.json $R/fs_envelope_cts25_b8811.json $R/fs_envelope_cts25_b4924.json \
    $SEL $SPL --prompts 100 --batch 32 \
    --output "$R/fs_env_eval_cts25_${SLURM_ARRAY_TASK_ID}.json"
else
  S=$((SLURM_ARRAY_TASK_ID - 14))
  python -u ruller-paper/experiments/rq3/lorare_task_metrics.py \
    --base /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf \
    --work /tmp/${USER}/fse_${SLURM_JOB_ID} --start 16 --prompts 34 --thresholds --ranks \
    --allocation $R/fs_envelope_lorare_b5880.json $R/fs_envelope_lorare_b3191.json $R/fs_envelope_lorare_b2683.json \
    --shard "$S" --shards 12 \
    --output "$R/fs_env_eval_lorare_s${S}.json"
fi
