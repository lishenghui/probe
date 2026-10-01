#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --array=0-11
#SBATCH --job-name=rq3-r0land
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_r0land_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" logs
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
TASKS=(wikisql gsm8k hellaswag glue_sst2 viggo glue_qqp glue_qnli glue_cola glue_mrpc conllpp e2e_nlg dbpedia)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
python -u ruller-paper/experiments/rq3/land_two_level_curve.py \
  --base "$BASE" --pool land --task "$TASK" \
  --work /tmp/${USER}/r0land_${SLURM_JOB_ID} \
  --examples 16 --example-start 200 --batch 4 --new 32 \
  --gamma 0 --layer-risk spectral --min-rank 0 --divergence output \
  --budget-step 8 --max-budget 256 \
  --output "artifacts/rq3/results/rank0_spectral_output_land12_${TASK}.json"
