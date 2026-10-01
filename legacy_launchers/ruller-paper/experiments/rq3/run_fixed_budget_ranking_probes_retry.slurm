#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --array=8-15
#SBATCH --job-name=rq3-fixrho2
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_fixedrho2_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" logs

I=$SLURM_ARRAY_TASK_ID
LAND=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2 gsm8k hellaswag viggo wikisql)
CTS=(task0112 task0149 task0223 task0260 task0297 task0334 task0371 task0408 task0593 task0852 task0889 task0926 task1037 task1111 task1370 task1407 task1444 task1518 task1592 task0704 task0778 task0815 task0963 task1148 task1333)
mapfile -t LORARE < <(for f in artifacts/rq3/results/layer_probe0_lorare_*.json; do jq -r .adapter "$f"; done | sort)
if (( I < 12 )); then
  T=${LAND[$I]}; POOL=land; BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
  OLD=artifacts/rq3/results/layer_probe0_${T}.json; OUT=artifacts/rq3/results/fixedrho_land_${T}.json
  BUDGETS=(832 1664 2496); EXTRA=(--example-start 200 --new 32)
elif (( I < 37 )); then
  T=${CTS[$((I-12))]}; POOL=cts; BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
  OLD=artifacts/rq3/results/layer_probe0_cts25_${T}.json; OUT=artifacts/rq3/results/fixedrho_cts_${T}.json
  BUDGETS=(2304 4608 6912); EXTRA=(--example-start 0 --new 32)
else
  T=${LORARE[$((I-37))]}; POOL=lorare; BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf
  OLD=artifacts/rq3/results/layer_probe0_lorare_${T}.json; OUT=artifacts/rq3/results/fixedrho_lorare_${T}.json
  BUDGETS=(1024 2048 3072); EXTRA=(--example-start 0 --new 48)
fi

python -u ruller-paper/experiments/rq3/layer_truncation_probe.py \
  --base "$BASE" --pool "$POOL" --task "$T" --work /tmp/${USER}/fixedrho_${SLURM_JOB_ID} \
  --examples 16 --batch 4 "${EXTRA[@]}" --skip-single-layer --random-allocations 0 \
  --fixed-budgets "${BUDGETS[@]}" --allocations-per-budget 20 \
  --merge-existing "$OLD" --output "$OUT"
