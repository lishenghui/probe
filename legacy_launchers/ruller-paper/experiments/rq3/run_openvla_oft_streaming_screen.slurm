#!/bin/bash
# Storage-bounded screening: one suite per array task, four variants sequentially.
# All checkpoints and rollout videos live on node-local /tmp. Only text logs and
# compression manifests are copied back to the project filesystem.
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=rq3-vlaoft-stream
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_vlaoft_stream_%A_%a.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
ENV=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
SUITES=(spatial object goal 10)
IDX=${SLURM_ARRAY_TASK_ID:?submit with --array=0-3}
SUITE=${SUITES[$IDX]}
REPO_ID="moojink/openvla-7b-oft-finetuned-libero-$SUITE"
JOB_ROOT="/tmp/${USER}/vlaoft_stream_${SLURM_JOB_ID}"
SOURCE="$JOB_ROOT/source"
VARIANTS="$JOB_ROOT/variants"
REPO="$JOB_ROOT/openvla-oft"
LIBERO="$JOB_ROOT/LIBERO"
VENV="$JOB_ROOT/env"
OUT="$W/artifacts/rq3/results/openvla-oft-screening/$SUITE"
mkdir -p "$JOB_ROOT" "$OUT"

export HF_HOME="$JOB_ROOT/hf"
export PIP_CACHE_DIR="$JOB_ROOT/pip-cache"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$JOB_ROOT/libero-config"
mkdir -p "$LIBERO_CONFIG_PATH"

cp -a "$W/artifacts/rq3/vendor/openvla-oft" "$REPO"
cp -a "$W/artifacts/rq3/vendor/LIBERO" "$LIBERO"
"$ENV/bin/hf" download "$REPO_ID" --local-dir "$SOURCE" \
  --exclude 'model-*.safetensors' --exclude 'model.safetensors.index.json'

"$ENV/bin/python" "$W/ruller-paper/experiments/rq3/prepare_openvla_oft_variants.py" \
  --source "$SOURCE" --output-root "$VARIANTS"

"$ENV/bin/python" -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
python -m pip install -q -e "$REPO" --no-deps
python -m pip install -q -e "$LIBERO" --no-deps
python -m pip install -q -r "$REPO/experiments/robot/libero/libero_requirements.txt"

for VARIANT in e100 e99 e95 e90; do
  CHECKPOINT="$VARIANTS/$VARIANT"
  VARIANT_OUT="$OUT/$VARIANT"
  mkdir -p "$VARIANT_OUT"

  cd "$REPO"
  python vla-scripts/merge_lora_weights_and_save.py \
    --base_checkpoint openvla/openvla-7b \
    --lora_finetuned_checkpoint_dir "$CHECKPOINT"

  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name "libero_$SUITE" \
    --num_trials_per_task 10 \
    --seed 7 \
    --center_crop True \
    --local_log_dir "$VARIANT_OUT" \
    --run_id_note "$SUITE-$VARIANT-seed7"

  if [[ -f "$CHECKPOINT/lora_adapter/compression.json" ]]; then
    cp "$CHECKPOINT/lora_adapter/compression.json" "$VARIANT_OUT/compression.json"
  fi
  # CHECKPOINT is explicitly node-local scratch. Remove only the just-produced
  # merged shards before building the next variant; LoRA/assets remain intact.
  find "$CHECKPOINT" -maxdepth 1 -type f \
    \( -name 'model-*.safetensors' -o -name 'model.safetensors.index.json' \) -delete
done
