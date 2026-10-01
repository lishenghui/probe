#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -J inst_libero
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/inst_libero_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/inst_libero_%j.err
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/nobackup/proj/disk/bloom/personal/shenghui/.uv_cache2
export TMPDIR=/tmp
LEROBOT=/nobackup/proj/disk/bloom/personal/shenghui/venvs/lerobot
uv pip install --python "$LEROBOT/bin/python" "lerobot[libero]==0.6.1" 2>&1 | tail -25
echo "=== verify ==="
"$LEROBOT/bin/python" - <<'PY'
import libero
print("libero OK", libero.__file__)
from lerobot.envs.libero import LiberoEnv
print("LiberoEnv OK")
from lerobot.scripts.lerobot_eval import eval_policy_all
print("lerobot_eval OK")
PY
echo "VERIFY DONE"
