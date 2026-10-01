#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --job-name=rq3-L12-safeM
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_land12_safeguarded_moderate_eval_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" logs
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
TASKS=(wikisql gsm8k hellaswag glue_sst2 viggo glue_qqp glue_qnli glue_cola glue_mrpc conllpp e2e_nlg dbpedia)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/landsafe_${SLURM_JOB_ID} \
  --tasks "$TASK" --thresholds --ranks \
  --allocation artifacts/rq3/results/safeguarded_land12_n128_k512_b2289.json \
  --examples 200 --batch 16 --no-margin-probe \
  --output "artifacts/rq3/results/safeguarded_moderate_eval_land12_${TASK}.json"
