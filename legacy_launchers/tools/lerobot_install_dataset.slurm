#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -J inst_ds
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/inst_ds_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/inst_ds_%j.err
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/nobackup/proj/disk/bloom/personal/shenghui/.uv_cache
export TMPDIR=/tmp
LEROBOT=/nobackup/proj/disk/bloom/personal/shenghui/venvs/lerobot
uv pip install --python "$LEROBOT/bin/python" "lerobot[dataset]==0.6.1" 2>&1 | tail -20
echo "=== verify ==="
"$LEROBOT/bin/python" - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print("LeRobotDataset OK")
from lerobot.scripts.lerobot_train import train
print("train OK")
import datasets; print("datasets", datasets.__version__)
try:
    import torchcodec; print("torchcodec OK")
except Exception as e:
    print("torchcodec unavailable:", type(e).__name__)
PY
echo "VERIFY DONE"
