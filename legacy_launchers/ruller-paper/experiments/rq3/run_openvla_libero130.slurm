#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --job-name=rq3-vla130
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_vla130_%x_%j.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
R_SRC="$W/artifacts/rq3/vendor/RLinf"
R="/tmp/${USER}/RLinf_${SLURM_JOB_ID}"
VENV="/tmp/${USER}/rlinf_${SLURM_JOB_ID}"
mkdir -p "/tmp/${USER}"
cp -a "$R_SRC" "$R"
# Each job installs LIBERO under its own node-local venv.  Keep LIBERO's
# absolute-path config node-local too: sharing ~/.libero/config.yaml lets
# concurrent jobs overwrite/truncate one another's config during startup.
export LIBERO_CONFIG_PATH="/tmp/${USER}/libero_config_${SLURM_JOB_ID}"
mkdir -p "$LIBERO_CONFIG_PATH"
export REPO_PATH="$R"
export EMBODIED_PATH="$R/examples/embodiment"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME="/tmp/${USER}/hf_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

cd "$R"
bash requirements/install.sh embodied --model openvla-oft --env libero \
  --venv "$VENV" --no-root --no-flash-attn --no-apex --install-rlinf
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1

MODEL="$W/artifacts/rq3/models/openvla-libero130-base"
RUN_LABEL="${RUN_LABEL:-$VARIANT}"
OUT="$W/artifacts/rq3/results/openvla130/$RUN_LABEL"
mkdir -p "$OUT"
ARGS=(
  "runner.only_eval=True"
  "runner.logger.log_path=$OUT"
  "rollout.model.model_path=$MODEL"
)
if [[ -n "${TOTAL_NUM_ENVS:-}" ]]; then
  ARGS+=("env.eval.total_num_envs=$TOTAL_NUM_ENVS")
fi
if [[ -n "${ROLLOUT_EPOCHS:-}" ]]; then
  ARGS+=("env.eval.rollout_epoch=$ROLLOUT_EPOCHS")
fi
if [[ -n "${MAX_EPISODE_STEPS:-}" ]]; then
  ARGS+=("env.eval.max_episode_steps=$MAX_EPISODE_STEPS" "env.eval.max_steps_per_rollout_epoch=$MAX_EPISODE_STEPS")
fi
if [[ "$VARIANT" != original ]]; then
  CORR="$W/artifacts/rq3/adapters/openvla130-corrections/$VARIANT"
  ARGS+=("rollout.model.is_lora=True" "rollout.model.lora_path=$CORR")
fi
bash evaluations/run_eval.sh libero_130_openvlaoft_eval "${ARGS[@]}"
