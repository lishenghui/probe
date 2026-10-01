#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:40:00
#SBATCH --array=0-13
#SBATCH --job-name=rq3-dbp
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_dbpedia_probe_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
CFGS=(equalL am1 am0.5 a0.25 a0.75 top4full cap4 k150 k173 k210 k232 k260 k300 k370)
C=${CFGS[$SLURM_ARRAY_TASK_ID]}
echo "config=$C"
python -u ruller-paper/experiments/rq3/predibase_task_metrics.py \
  --base "$BASE" --work /tmp/${USER}/dbp_${SLURM_JOB_ID} --tasks dbpedia \
  --thresholds --ranks \
  --allocation "artifacts/rq3/results/dbpedia_probe/cfg_${C}.json" \
  --examples 200 --batch 16 --no-margin-probe \
  --output "artifacts/rq3/results/dbpedia_probe/res_${C}.json"
