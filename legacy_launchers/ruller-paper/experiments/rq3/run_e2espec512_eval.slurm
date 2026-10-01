#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --job-name=rq3-spec512
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_spec512_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
R=artifacts/rq3/results; A=$R/fra_clean_alloc
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
TASKS=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2 gsm8k hellaswag viggo wikisql)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
# The spectral rung re-solved on curves that reach full rank. The earlier run
# stopped at k=256, which two adapters wanted to exceed at the moderate budget,
# so that rung was competing under a ceiling the other rungs did not have.
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/spec512_${SLURM_JOB_ID} --tasks "$T" \
  --thresholds --ranks \
  --allocation $A/land_e2e_spec512_b2289.json $A/land_e2e_spec512_b1321.json $A/land_e2e_spec512_b943.json \
  --examples 200 --batch 16 --no-margin-probe \
  --output "$R/spec512_eval_land_${T}.json"
