#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J rlinf_eval_all
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rlinf_eval_all_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rlinf_eval_all_%j.err

set -uo pipefail
RLINF=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/RLinf
ROOT=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/artifacts/rq3/models/rlinf_tmp
OUT=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/artifacts/rq3/results/rlinf_eval_all
export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=0
mkdir -p "$ROOT" "$OUT"

cd "$RLINF"
source .venv/bin/activate
PY=$(which python)
echo "arch=$(uname -m) host=$(hostname) python=$PY"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

PATTERNS='["config.json","configuration_prismatic.py","modeling_prismatic.py","processing_prismatic.py","dataset_statistics.json","generation_config.json","preprocessor_config.json","processor_config.json","model-0000*-of-0000*.safetensors","model.safetensors.index.json","tokenizer.json","tokenizer_config.json","tokenizer.model","special_tokens_map.json","added_tokens.json"]'

run_one () {
  local repo="$1" cfg="$2" nenvs="$3" epochs="$4" tag="$5"
  local D="$ROOT/$tag"
  echo ""
  echo "############################################################"
  echo "### $tag : $repo  (cfg=$cfg envs=$nenvs epochs=$epochs)"
  echo "############################################################"
  rm -rf "$D"
  "$PY" - "$repo" "$D" "$PATTERNS" <<'PY'
import sys, json
from huggingface_hub import snapshot_download
repo, dest, pats = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=pats, max_workers=8)
print("downloaded", dest)
PY
  if [ ! -f "$D/dataset_statistics.json" ]; then echo "MISSING dataset_statistics for $tag"; return 1; fi
  local UK
  UK=$("$PY" -c "import json;print(list(json.load(open('$D/dataset_statistics.json')).keys())[0])")
  echo "unnorm_key=$UK"
  bash evaluations/run_eval.sh libero "$cfg" \
    rollout.model.model_path="$D" \
    rollout.model.unnorm_key="$UK" \
    rollout.model.attn_implementation=sdpa \
    'runner.logger.logger_backends=[]' \
    env.eval.rollout_epoch="$epochs" \
    env.eval.total_num_envs="$nenvs" \
    env.eval.video_cfg.save_video=False \
    > "$OUT/$tag.log" 2>&1
  echo "--- $tag metric ---"
  grep -aE "success_once=|success_at_end=|num_trajectories=" "$OUT/$tag.log" | tail -3
  cp -f "$OUT/$tag.log" "$OUT/$tag.done" 2>/dev/null || true
  rm -rf "$D"
  df -h /nobackup/proj/disk/bloom/personal/shenghui/ | tail -1
}

run_one RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial libero_spatial_openvlaoft_eval 20 2 grpo_spatial
run_one RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object  libero_object_openvlaoft_eval  20 2 grpo_object
run_one RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal    libero_goal_openvlaoft_eval    20 2 grpo_goal
run_one RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-long    libero_10_openvlaoft_eval      20 2 grpo_long
run_one RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90      libero_90_openvlaoft_eval      90 1 grpo_90

echo ""
echo "================= SUMMARY ================="
for t in grpo_spatial grpo_object grpo_goal grpo_long grpo_90; do
  printf "%-14s " "$t"; grep -aE "success_once=" "$OUT/$t.log" 2>/dev/null | tail -1
done
echo "=== ALL DONE ==="
