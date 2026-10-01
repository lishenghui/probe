#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --array=0-24
#SBATCH --job-name=rq3-ctsprobe
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_cts25_layer_probe_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
ALL=(task0112 task0149 task0223 task0260 task0297 task0334 task0371 task0408 task0593 task0852 task0889 task0926 task1037 task1111 task1370 task1407 task1444 task1518 task1592 task0704 task0778 task0815 task0963 task1148 task1333)
T=${ALL[$SLURM_ARRAY_TASK_ID]}
echo "task=$T"
# 96 modules x 15 ranks = 1440 single-layer measurements, three times LoRA Land's,
# so the walltime is generous. Calibration prompts are rows 0-15 of the train
# split, matching this pool's curve protocol and disjoint from its test split.
python -u ruller-paper/experiments/rq3/layer_truncation_probe.py \
  --base "$BASE" --pool cts --task "$T" --work /tmp/${USER}/ctsprobe_${SLURM_JOB_ID} \
  --examples 16 --example-start 0 --batch 4 --new 32 --random-allocations 40 \
  --output "artifacts/rq3/results/layer_probe_cts25_${T}.json"
