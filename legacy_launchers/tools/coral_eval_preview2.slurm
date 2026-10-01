#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -t 01:00:00
#SBATCH -J coral_eval2
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/coral_eval2_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/coral_eval2_%j.err

set -uo pipefail
CORAL=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/CORAL
LIBERO_ROOT=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/LIBERO
SIMVLA=/nobackup/proj/disk/bloom/personal/shenghui/venvs/simvla
LIBERO_VENV=/nobackup/proj/disk/bloom/personal/shenghui/venvs/libero

export PATH="$SIMVLA/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$CORAL:$LIBERO_ROOT"
export PORT=8093
export NUM_TRIALS=${NUM_TRIALS:-20}

OUT="$CORAL/eval_preview2_$(date +%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "arch=$(uname -m) host=$(hostname) out=$OUT"

run_condition() {
  local cond="$1" lora_dir="$2"
  echo ""
  echo "############################################################"
  echo "### CONDITION: $cond   (lora_dir=$lora_dir)"
  echo "############################################################"

  CUDA_VISIBLE_DEVICES=0 $SIMVLA/bin/python -u "$CORAL/serve_smolvlm_coral.py" \
    --model "$CORAL/pretrained/SimVLA-LIBERO" \
    --lora-dir "$lora_dir" \
    --task-index "$CORAL/datasets/coral_metas/task_index.json" \
    --norm-stats "$CORAL/norm_stats/libero_norm.json" \
    --port $PORT > "$OUT/server_${cond}.log" 2>&1 &
  local SRV=$!

  echo "waiting for server (pid $SRV)..."
  local ok=0
  for i in $(seq 1 120); do
    if grep -q "Connect at:" "$OUT/server_${cond}.log" 2>/dev/null; then ok=1; break; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED"; tail -30 "$OUT/server_${cond}.log"; return 1; fi
    sleep 2
  done
  [ $ok -eq 1 ] || { echo "server timeout"; tail -20 "$OUT/server_${cond}.log"; kill $SRV 2>/dev/null; return 1; }
  echo "server ready"; sleep 5

  for spec in "libero_spatial:2:spatial_t2" "libero_goal:8:goal_t8"; do
    local suite="${spec%%:*}"; local rest="${spec#*:}"
    local tid="${rest%%:*}"; local tag="${rest##*:}"
    echo "--- eval $cond / $suite task $tid ---"
    CUDA_VISIBLE_DEVICES=1 $LIBERO_VENV/bin/python -u \
      "$CORAL/evaluation/libero/libero_client.py" \
      --host 127.0.0.1 --port $PORT --client_type websocket \
      --task_suite "$suite" --task_ids $tid \
      --num_trials $NUM_TRIALS --no_video --video_out "$OUT" \
      > "$OUT/${cond}_${tag}.txt" 2>&1
    grep -E "Total success rate|Task .*:|Ep .*SUCCESS|Ep .*FAILURE" "$OUT/${cond}_${tag}.txt" | tail -3
  done

  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true
  echo "condition $cond done"
}

mkdir -p "$CORAL/lora_adapters/__empty__"
run_condition base "$CORAL/lora_adapters/__empty__"
run_condition expert "$CORAL/lora_adapters/coral_r16_preview_1gpu"

echo ""
echo "=================== SUMMARY ==================="
for f in "$OUT"/*.txt; do
  echo "--- $(basename $f) ---"
  grep -E "Total success rate|Task [0-9]+:" "$f" 2>/dev/null | tail -4
done
echo "Full logs: $OUT"
