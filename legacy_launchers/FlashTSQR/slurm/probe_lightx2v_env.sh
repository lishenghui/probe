#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=probe_lightx2v
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/probe_lightx2v_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
PROBE=/tmp/lightx2v_probe_${SLURM_JOB_ID}
git clone --depth 1 https://github.com/ModelTC/LightX2V.git "$PROBE"
export PYTHONPATH="$PROBE:${PYTHONPATH:-}"
df -h /tmp
python - <<'PY'
import importlib.util
modules = ['torch','torchvision','diffusers','transformers','tokenizers','accelerate','cv2',
           'imageio','einops','loguru','omegaconf','peft','ftfy','decord','av','flash_attn']
for module in modules:
    print(module, bool(importlib.util.find_spec(module)))
import lightx2v
print('LightX2V import OK', lightx2v.__version__)
PY
