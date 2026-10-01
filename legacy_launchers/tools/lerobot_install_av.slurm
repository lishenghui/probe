#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -J inst_av
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/inst_av_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/inst_av_%j.err
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/nobackup/proj/disk/bloom/personal/shenghui/.uv_cache
export TMPDIR=/tmp
LEROBOT=/nobackup/proj/disk/bloom/personal/shenghui/venvs/lerobot
uv pip install --python "$LEROBOT/bin/python" av 2>&1 | tail -4
"$LEROBOT/bin/python" -c 'import av; print("av", av.__version__)'
"$LEROBOT/bin/python" -c 'from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print("SmolVLA OK")'
"$LEROBOT/bin/python" -c 'from lerobot.datasets.lerobot_dataset import LeRobotDataset; print("LeRobotDataset OK")'
"$LEROBOT/bin/python" -c 'from lerobot.scripts.lerobot_train import train; print("train OK")'
echo "VERIFY DONE"
