#!/bin/bash
# Official closed-loop screening: 4 suites x 4 variants, 10 trials/task.
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=rq3-vlaoft-screen
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_vlaoft_screen_%A_%a.out
set -euo pipefail

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
ENV=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
REPO_SRC="$W/artifacts/rq3/vendor/openvla-oft"
LIBERO_SRC="$W/artifacts/rq3/vendor/LIBERO"
ROOT="$W/artifacts/rq3/models/openvla-oft-variants"
OUT_ROOT="$W/artifacts/rq3/results/openvla-oft-screening"
PY_OVERLAY="$W/artifacts/rq3/python_overlays/openvla_oft"
SUITES=(spatial object goal 10)
VARIANTS=(e100 e99 e95 e90)
IDX=${SLURM_ARRAY_TASK_ID:?submit with --array=0-15}
SUITE=${SUITES[$((IDX / 4))]}
VARIANT=${VARIANTS[$((IDX % 4))]}
NUM_TRIALS=${NUM_TRIALS:-10}
CHECKPOINT="$ROOT/$SUITE/$VARIANT"
OUT="$OUT_ROOT/$SUITE/$VARIANT"

test -f "$CHECKPOINT/.merge_complete"
mkdir -p "/tmp/${USER}" "$OUT"
REPO="/tmp/${USER}/openvla_oft_${SLURM_JOB_ID}"
LIBERO="/tmp/${USER}/libero_${SLURM_JOB_ID}"
VENV="/tmp/${USER}/openvla_oft_env_${SLURM_JOB_ID}"
cp -a "$REPO_SRC" "$REPO"
cp -a "$LIBERO_SRC" "$LIBERO"
"$ENV/bin/python" -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
export PYTHONPATH="$PY_OVERLAY:$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import draccus, peft; assert peft.__version__ == "0.11.1"; from transformers import AutoModelForVision2Seq'
python -m pip install -q -e "$REPO" --no-deps
python -m pip install -q -e "$LIBERO" --no-deps
python -m pip install -q -r "$REPO/experiments/robot/libero/libero_requirements.txt"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME="$W/artifacts/rq3/hf_cache_openvla_oft"
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="/tmp/${USER}/libero_config_${SLURM_JOB_ID}"
mkdir -p "$LIBERO_CONFIG_PATH"

cd "$REPO"
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CHECKPOINT" \
  --task_suite_name "libero_$SUITE" \
  --num_trials_per_task "$NUM_TRIALS" \
  --seed 7 \
  --center_crop True \
  --local_log_dir "$OUT" \
  --run_id_note "$SUITE-$VARIANT-seed7"
