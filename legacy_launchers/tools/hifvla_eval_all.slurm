#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -J hifvla_all
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/hifvla_all_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/hifvla_all_%j.err
set -uo pipefail
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
VENV=$W/third_party/RLinf/.venv
HIF=$W/third_party/HiF-VLA
export PATH="$W/tools/bin:$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HIF:$W/third_party/LIBERO"
export HF_HOME=$W/artifacts/rq3/hf_cache_openvla_oft
cd "$HIF"
echo "host=$(hostname) ffmpeg=$(which ffmpeg)"

run_one () {
  local suite="$1" tsuite="$2" unnorm="$3"
  local ck="ckpt/hifvla-libero-${suite}"
  echo ""; echo "##################### $suite ($tsuite) #####################"
  rm -rf "$ck"
  "$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='minnielin/hifvla-libero-${suite}', local_dir='$ck', max_workers=8)
print('downloaded')" || { echo "DL FAIL $suite"; return 1; }
  "$VENV/bin/python" experiments/robot/libero/run_libero_eval.py \
    --use_proprio True --num_images_in_input 2 --use_film False \
    --pretrained_checkpoint "$ck" --task_suite_name "$tsuite" \
    --history_length 8 --num_trials_per_task ${TRIALS:-5} --unnorm_key "$unnorm" \
    > "$W/artifacts/rq3/results/hifvla_${suite}.log" 2>&1
  echo "--- $suite ---"; grep -aA1 -iE "Overall success rate" "$W/artifacts/rq3/results/hifvla_${suite}.log" | tail -2
  rm -rf "$ck"
  df -h /nobackup/proj/disk/bloom/personal/shenghui/ | tail -1
}
mkdir -p "$W/artifacts/rq3/results"

run_one object libero_object libero_object_no_noops
run_one goal   libero_goal   libero_goal_no_noops
run_one long   libero_10     libero_10_no_noops

echo ""; echo "================= SUMMARY ================="
for s in spatial object goal long; do
  f="$W/artifacts/rq3/results/hifvla_${s}.log"
  [ -f "$f" ] && printf "%-10s " "$s" && grep -aA1 -iE "Overall success rate" "$f" | tail -1
done
echo "=== ALL DONE ==="
