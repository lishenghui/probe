#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --job-name=rq3-jacsur
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_jacobian_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
TASKS=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2 gsm8k hellaswag viggo wikisql)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
echo "task=$T"
# M=8 is the reported setting; q is stored per projection so M=1,2,4 are exact
# prefixes of the same run and need no extra compute.
python -u ruller-paper/experiments/rq3/jacobian_layer_surrogate.py \
  --base "$BASE" --task "$T" --work /tmp/${USER}/jac_${SLURM_JOB_ID} \
  --examples 16 --example-start 200 --batch 2 --projections 8 \
  --output "artifacts/rq3/results/jacobian_surrogate_${T}.json"
