#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -J smolvla_eval_spatial
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/smolvla_eval_spatial_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/smolvla_eval_spatial_%j.err

set -uo pipefail
LEROBOT=/nobackup/proj/disk/bloom/personal/shenghui/venvs/lerobot
SMOLVLA=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/CORAL/smolvla
export PATH="$LEROBOT/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME="$SMOLVLA/hf_home"
export HF_LEROBOT_HOME="$SMOLVLA/lerobot_home"
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

BASE="$SMOLVLA/pretrained/smolvla_libero"
EXPERT="$SMOLVLA/outputs/spatial_t2_r16/checkpoints/002000/pretrained_model"
N=${N:-10}
OUT="$SMOLVLA/eval/spatial_t2"
mkdir -p "$OUT"

echo "arch=$(uname -m) host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

run_eval () {
  local tag="$1" policy="$2"
  echo ""
  echo "############ EVAL $tag : task libero_spatial[2] n=$N ############"
  lerobot-eval \
    --policy.path="$policy" \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids="[2]" \
    --eval.batch_size=10 \
    --eval.n_episodes=$N \
    --env.max_parallel_tasks=1 \
    --output_dir="$OUT/$tag" 2>&1 | tail -40
}

run_eval base "$BASE"
run_eval expert_r16 "$EXPERT"

echo "=== EVAL DONE ==="
