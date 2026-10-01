#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -J rlinf_eval_recon64
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rlinf_eval_recon64_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rlinf_eval_recon64_%j.err

set -uo pipefail
RLINF=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/RLinf
MODEL=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/artifacts/rq3/models/rlinf-libero130-rl-recon64
export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1

cd "$RLINF"
source .venv/bin/activate
echo "arch=$(uname -m) host=$(hostname) python=$(which python)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
python -c "import torch, rlinf, libero, robosuite; print('imports ok torch', torch.__version__)"

bash evaluations/run_eval.sh libero libero_object_openvlaoft_eval \
  rollout.model.model_path="$MODEL" \
  rollout.model.unnorm_key=libero_130_no_noops_trajall \
  rollout.model.attn_implementation=sdpa \
  'runner.logger.logger_backends=[]' \
  env.eval.rollout_epoch=2 \
  env.eval.total_num_envs=10 \
  env.eval.max_episode_steps=280 \
  env.eval.max_steps_per_rollout_epoch=280 \
  env.eval.video_cfg.save_video=False

echo "=== RLINF EVAL EXIT $? ==="
