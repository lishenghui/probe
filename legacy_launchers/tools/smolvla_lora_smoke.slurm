#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -J smolvla_lora_smoke
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/smolvla_lora_smoke_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/smolvla_lora_smoke_%j.err

set -uo pipefail
LEROBOT=/nobackup/proj/disk/bloom/personal/shenghui/venvs/lerobot
SMOLVLA=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/CORAL/smolvla
export PATH="$LEROBOT/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME="$SMOLVLA/hf_home"
export HF_LEROBOT_HOME="$SMOLVLA/lerobot_home"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

BASE="$SMOLVLA/pretrained/smolvla_libero"
DS="$SMOLVLA/datasets/libero"
OUT="$SMOLVLA/outputs/spatial_t2_r16_smoke"
EPISODES="[1283,1287,1289,1299,1310,1315,1316,1317,1320,1345,1353,1354,1358,1377,1380,1383,1385,1426,1428,1432,1440,1442,1445,1455,1460,1466,1478,1479,1483,1509,1510,1514,1545,1549,1556,1557,1573,1603,1622,1641,1652,1657,1665,1668,1682,1686]"

echo "arch=$(uname -m) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
"$LEROBOT/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

lerobot-train \
  --policy.path="$BASE" \
  --policy.device=cuda \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.optimizer_lr=1e-3 \
  --policy.scheduler_decay_lr=1e-4 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root="$DS" \
  --dataset.episodes="$EPISODES" \
  --dataset.video_backend=pyav \
  --steps=50 \
  --batch_size=16 \
  --save_freq=50 \
  --num_workers=4 \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --output_dir="$OUT" \
  --job_name=smolvla_spatial_t2_r16_smoke

echo "=== SMOKE DONE ==="
find "$OUT" -name "*.safetensors" -o -name "adapter*" 2>/dev/null | head
