#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --job-name=rq3-vlacap
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_vlacap_%j.out
set -euo pipefail
# Short original-policy rollout whose only purpose is to record observations for
# same-input probing.  Two rollout epochs is plenty: the probe needs a few
# hundred (observation, action) pairs, not a success-rate estimate.
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
R="/tmp/${USER}/RLinf_${SLURM_JOB_ID}"
VENV="/tmp/${USER}/rlinf_${SLURM_JOB_ID}"
mkdir -p "/tmp/${USER}"
cp -a "$W/artifacts/rq3/vendor/RLinf" "$R"

# Append the probe to the vendored copy rather than editing the repo's source.
MODEL_PY="$R/rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py"
cat "$W/ruller-paper/experiments/rq3/vla_probe_patch.py" >> "$MODEL_PY"
echo "patched $MODEL_PY"

export LIBERO_CONFIG_PATH="/tmp/${USER}/libero_config_${SLURM_JOB_ID}"
mkdir -p "$LIBERO_CONFIG_PATH"
export REPO_PATH="$R" EMBODIED_PATH="$R/examples/embodiment"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME="/tmp/${USER}/hf_${SLURM_JOB_ID}" TOKENIZERS_PARALLELISM=false
cd "$R"
bash requirements/install.sh embodied --model openvla-oft --env libero \
  --venv "$VENV" --no-root --no-flash-attn --no-apex --install-rlinf
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1

export RQ3_DUMP_DIR="$W/artifacts/rq3/results/openvla130/observations"
export RQ3_DUMP_LIMIT="${RQ3_DUMP_LIMIT:-64}"
rm -rf "$RQ3_DUMP_DIR"; mkdir -p "$RQ3_DUMP_DIR"

OUT="$W/artifacts/rq3/results/openvla130/capture"
mkdir -p "$OUT"
bash evaluations/run_eval.sh libero_130_openvlaoft_eval \
  "runner.only_eval=True" \
  "runner.logger.log_path=$OUT" \
  "rollout.model.model_path=$W/artifacts/rq3/models/openvla-libero130-base" \
  "env.eval.total_num_envs=16" \
  "env.eval.rollout_epoch=2"
echo "captured: $(ls "$RQ3_DUMP_DIR" | wc -l) batches"
du -sh "$RQ3_DUMP_DIR"
