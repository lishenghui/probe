#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --array=0-43
#SBATCH --job-name=rq3-L12b23
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_land12_b2289_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
TASKS=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_sst2 gsm8k hellaswag viggo wikisql)
RULES=(a0.5 a1.0 a1.5 equal)
T=${TASKS[$((SLURM_ARRAY_TASK_ID % 11))]}
G=${RULES[$((SLURM_ARRAY_TASK_ID / 11))]}
echo "task=$T rule=$G"
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/l12b23_${SLURM_JOB_ID} --tasks "$T" \
  --thresholds --ranks \
  --allocation "artifacts/rq3/results/layer_land12_b2289_${G}.json" \
  --examples 200 --batch 16 --no-margin-probe \
  --output "artifacts/rq3/results/land12_lay2289_${G}_${T}.json"
