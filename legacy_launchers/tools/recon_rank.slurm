#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -J recon64
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/recon64_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/recon64_%j.err
set -uo pipefail
VENV=/nobackup/proj/disk/bloom/personal/shenghui/venvs/simvla
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
"$VENV/bin/python" -u "$W/tools/recon_rank.py" \
  "$W/artifacts/rq3/models/openvla-7b-base" \
  "$W/artifacts/rq3/models/rlinf-libero130-rl" \
  "$W/artifacts/rq3/models/rlinf-libero130-rl-recon64" 64
