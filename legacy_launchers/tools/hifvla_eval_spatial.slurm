#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -J hifvla_eval
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/hifvla_eval_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/hifvla_eval_%j.err
set -uo pipefail
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
VENV=$W/third_party/RLinf/.venv
HIF=$W/third_party/HiF-VLA
LIB=$W/artifacts/rq3/vendor/openvla-oft
FFBASE=$W/venvs/lerobot/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2

# ffmpeg on PATH (shared symlink to imageio-ffmpeg bundled binary)
export PATH="$W/tools/bin:$HOME/.local/bin:$PATH"
# ensure av present
"$VENV/bin/python" -c "import av" 2>/dev/null || \
  { export UV_CACHE_DIR=$W/.uv_cache2; uv pip install --python "$VENV/bin/python" av 2>&1 | tail -2; }

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HIF:$W/third_party/LIBERO"
export HF_HOME=$W/artifacts/rq3/hf_cache_openvla_oft
cd "$HIF"
echo "arch=$(uname -m) host=$(hostname) python=$($VENV/bin/python --version) ffmpeg=$(which ffmpeg)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

"$VENV/bin/python" experiments/robot/libero/run_libero_eval.py \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --pretrained_checkpoint ckpt/hifvla-libero-spatial \
  --task_suite_name libero_spatial \
  --history_length 8 \
  --num_trials_per_task ${TRIALS:-1} \
  --unnorm_key libero_spatial_no_noops
echo "=== HIFVLA EXIT $? ==="
