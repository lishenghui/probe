#!/usr/bin/env bash
# Max frames / batch with the whole fleet resident under a GPU memory cap. Submit from repo root.
# Env: PROBE_PYTHON, FLEET_DIR, MODEL_DIR, COMPRESSED_DIR, CAP_GIB; optional FP8=1, VARIANTS, PROBE_SCRATCH.
#SBATCH --job-name=probe-fleet-capacity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
set -euo pipefail
source "$HOME/.config/hpc-agent/bootstrap.sh"
: "${PROBE_PYTHON:?}" "${FLEET_DIR:?}" "${MODEL_DIR:?}" "${COMPRESSED_DIR:?}" "${CAP_GIB:?}"
cd "$SLURM_SUBMIT_DIR"
SCR=${PROBE_SCRATCH:-/nobackup/proj/disk/bloom/personal/shenghui/scratch}
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TMPDIR="$SCR/probe-fleet-capacity-$SLURM_JOB_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PROBE_PYTHON" -u tools/bench_fleet_capacity.py --model-dir "$MODEL_DIR" --fleet-dir "$FLEET_DIR" \
  --compressed-dir "$COMPRESSED_DIR" --image "$MODEL_DIR/diffusers_config/examples/i2v_input.JPG" \
  --prompts configs/remade_pilot_prompts.json --cap-gib "$CAP_GIB" ${FP8:+--fp8-storage} \
  ${VARIANTS:+--variants $VARIANTS} --output "artifacts/fleet_capacity/$SLURM_JOB_ID/capacity.json"
