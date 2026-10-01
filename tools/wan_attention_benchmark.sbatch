#!/usr/bin/env bash
# Do not install the `kernels` package into the shared env: kernels 0.17 breaks transformers 5.5 imports.
# Times short no-LoRA Wan I2V generations per attention backend. Submit from repo root.
# Env: PROBE_PYTHON, MODEL_DIR, PLAN; optional PROBE_SCRATCH, BACKENDS, STEPS.
#SBATCH --job-name=probe-wan-attn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:45:00
set -euo pipefail
source "$HOME/.config/hpc-agent/bootstrap.sh"
: "${PROBE_PYTHON:?}" "${MODEL_DIR:?}" "${PLAN:?}"
cd "$SLURM_SUBMIT_DIR"
SCR=${PROBE_SCRATCH:-/nobackup/proj/disk/bloom/personal/shenghui/scratch}
export PYTHONDONTWRITEBYTECODE=1 PIP_CACHE_DIR="$SCR/pip_cache" TMPDIR="$SCR/probe-wan-attn-$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR"
"$PROBE_PYTHON" -u tools/generate_wan_i2v_fleet_pilot.py --model-dir "$MODEL_DIR" --plan "$PLAN" \
  --image "$MODEL_DIR/diffusers_config/examples/i2v_input.JPG" \
  --output-dir "artifacts/wan_attention_benchmark/$SLURM_JOB_ID" --steps "${STEPS:-5}" \
  --benchmark-backends ${BACKENDS:-native _native_cudnn _native_efficient _native_flash}
