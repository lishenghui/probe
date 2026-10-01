#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -J smolvla_r64
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/smolvla_r64_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/smolvla_r64_%j.err

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

TASK=${TASK:?set TASK}
EPISODES=${EPISODES:?set EPISODES}
RANK=${RANK:-64}
ALPHA=${ALPHA:-64}
STEPS=${STEPS:-10000}
BATCH=${BATCH:-32}
OUT="$SMOLVLA/outputs/${TASK}_r${RANK}"

echo "arch=$(uname -m) host=$(hostname) task=$TASK r=$RANK alpha=$ALPHA steps=$STEPS batch=$BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

lerobot-train \
  --policy.path="$BASE" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.optimizer_lr=1e-3 \
  --policy.scheduler_decay_lr=1e-4 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root="$DS" \
  --dataset.episodes="$EPISODES" \
  --dataset.video_backend=pyav \
  --steps="$STEPS" \
  --batch_size="$BATCH" \
  --save_freq=2500 \
  --num_workers=4 \
  --peft.method_type=LORA \
  --peft.r="$RANK" \
  --peft.lora_alpha="$ALPHA" \
  --output_dir="$OUT" \
  --job_name="smolvla_${TASK}_r${RANK}"

echo "=== TRAIN DONE $TASK r$RANK ==="
find "$OUT" -name "adapter_model.safetensors" -exec ls -la {} \;
