#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -J fixmeta
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/fixmeta_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/fixmeta_%j.err
set -uo pipefail
VENV=/nobackup/proj/disk/bloom/personal/shenghui/venvs/simvla
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
"$VENV/bin/python" -u "$W/tools/fix_safetensors_meta.py" "$W/artifacts/rq3/models/rlinf-libero130-rl-recon64"
