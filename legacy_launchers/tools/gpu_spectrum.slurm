#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -J gpu_spectrum
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/gpu_spectrum_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/gpu_spectrum_%j.err
set -uo pipefail
VENV=/nobackup/proj/disk/bloom/personal/shenghui/venvs/simvla
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
"$VENV/bin/python" -u "$W/tools/gpu_spectrum.py" \
  "$W/artifacts/rq3/models/openvla-7b-base" \
  "$W/artifacts/rq3/models/rlinf-libero130-rl"
