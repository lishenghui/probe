#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=rq3-cttop
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_cts_task_topup_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
# Same quota workaround as the original shards: Lustre reports the project quota
# as the filesystem size, so shutil sees 0 bytes free and datasets refuses to
# download. HF_HOME goes to node-local scratch; the base model is read from the
# shared staging path, which needs no space check.
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME"
export STAGED_INSTRUCT=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
# The eight adapters the original sweep skipped. Three more (task0630, task1481,
# task0075) have no dataset on the hub at all, so 30 is this pool's ceiling.
# task0741 and task0667 stay below --min-prompts 30 even with valid merged in and
# are expected to skip themselves; they are listed so the log records why.
python -u ruller-paper/experiments/rq3/cts_task_metrics.py \
  --base "$STAGED_INSTRUCT" \
  --strengths artifacts/rq3/results/cts_adapter_strength.json \
  --work /tmp/${USER}/cts_topup_${SLURM_JOB_ID} \
  --adapters task0667 task0704 task0741 task0778 task0815 task0963 task1148 task1333 \
  --splits test valid --min-prompts 30 --prompts 100 --batch 8 \
  --thresholds 0.99 0.95 0.90 0.80 0.70 0.50 --ranks 2 4 6 8 10 12 \
  --output artifacts/rq3/results/cts_task_topup.json
