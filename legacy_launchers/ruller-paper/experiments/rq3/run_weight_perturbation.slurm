#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=rq3-perturb
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_perturb_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
cd "$W"
S=ruller-paper/experiments/rq3/measure_weight_perturbation.py
A=artifacts/rq3/adapters
OUT=artifacts/rq3/results/weight_perturbation.json
rm -f "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader
python -u $S --family "mbpp/code-gen"   --original $A/mbpp-qwen25/original \
  --variant $A/mbpp-qwen25/e99 $A/mbpp-qwen25/e95 $A/mbpp-qwen25/e90 --output $OUT
python -u $S --family "editscore/VLM"   --original $A/editscore/original \
  --variant $A/editscore/e99 $A/editscore/e95 $A/editscore/e90 --output $OUT
python -u $S --family "videogpa/video"  --original $A/videogpa-i2v-original \
  --variant $A/videogpa-i2v-compressed/e99 $A/videogpa-i2v-compressed/e95 $A/videogpa-i2v-compressed/e90 --output $OUT
python -u $S --family "openvla/VLA"     --original $A/openvla130-original \
  --variant $A/openvla130-corrections/e99 $A/openvla130-corrections/e95 $A/openvla130-corrections/e90 \
  --correction-self-reference --output $OUT
echo "RESULT=$OUT"
