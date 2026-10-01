#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --job-name=rq3-localdiv
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_localdiv_%j.out
set -uo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
E=ruller-paper/experiments/rq3
A=artifacts/rq3/adapters
M=artifacts/rq3/models
R=artifacts/rq3/results
# Datasets must not live on the project disk: Lustre reports the project
# quota as the filesystem size, so shutil.disk_usage sees 0 bytes free and
# datasets refuses to download even though writes would succeed. Models are
# read from the shared staging path instead, which needs no space check.
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME"
export STAGED_V01=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
export STAGED_INSTRUCT=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
nvidia-smi --query-gpu=name --format=csv,noheader

echo "############ VIDEO (CogVideoX velocity divergence) ############"
python -u $E/probe_local_divergence_video.py \
  --model $M/CogVideoX-5B-I2V \
  --original $A/videogpa-i2v-original \
  --variant $A/videogpa-i2v-compressed/e99 $A/videogpa-i2v-compressed/e95 $A/videogpa-i2v-compressed/e90 \
  --output $R/local_divergence_video.json

echo; echo "############ VLM (Qwen3-VL teacher-forced JS) ############"
python -u $E/probe_local_divergence_lm.py \
  --model $M/qwen3-vl-4b-instruct \
  --original $A/editscore/original \
  --variant $A/editscore/e99 $A/editscore/e95 $A/editscore/e90 \
  --prompts $R/editscore_prompts.jsonl \
  --output $R/local_divergence_vlm.json
