#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=00:40:00
#SBATCH --array=0-36
#SBATCH --job-name=rq3-fdp0curve
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_fdp0curve_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
I=$SLURM_ARRAY_TASK_ID
if [ "$I" -lt 12 ]; then
  TASKS=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2 gsm8k hellaswag viggo wikisql)
  T=${TASKS[$I]}; POOL=land
  BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
  COST="artifacts/rq3/results/layer_probe0_${T}.json"; OUT="artifacts/rq3/results/functional_dp0_output_land12_${T}.json"
  EXTRA=(--example-start 200 --new 32 --budget-step 8 --max-budget 512)
else
  TASKS=(task0112 task0149 task0223 task0260 task0297 task0334 task0371 task0408 task0593 task0852 task0889 task0926 task1037 task1111 task1370 task1407 task1444 task1518 task1592 task0704 task0778 task0815 task0963 task1148 task1333)
  T=${TASKS[$((I-12))]}; POOL=cts
  BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
  COST="artifacts/rq3/results/layer_probe0_cts25_${T}.json"; OUT="artifacts/rq3/results/functional_dp0_output_cts25_${T}.json"
  EXTRA=(--example-start 0 --new 32 --budget-step 16 --max-budget 1536)
fi
python -u ruller-paper/experiments/rq3/land_two_level_curve.py \
  --base "$BASE" --pool "$POOL" --task "$T" \
  --work /tmp/${USER}/fdp0curve_${SLURM_JOB_ID} \
  --examples 16 --batch 4 "${EXTRA[@]}" --divergence output \
  --layer-risk functional --min-rank 0 --layer-costs "$COST" --output "$OUT"
