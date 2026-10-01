#!/usr/bin/env bash
# PEFT-free multi-adapter LoRA: correctness, batch scaling, residency and swap. Submit from repo root.
# Env: PROBE_PYTHON, FLEET_DIR, MODEL_DIR, COMPRESSED_DIR; optional PROBE_SCRATCH.
#SBATCH --job-name=probe-multi-lora
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
set -euo pipefail
source "$HOME/.config/hpc-agent/bootstrap.sh"
: "${PROBE_PYTHON:?}" "${FLEET_DIR:?}" "${MODEL_DIR:?}" "${COMPRESSED_DIR:?}"
cd "$SLURM_SUBMIT_DIR"
SCR=${PROBE_SCRATCH:-/nobackup/proj/disk/bloom/personal/shenghui/scratch}
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TMPDIR="$SCR/probe-multi-lora-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PROBE_PYTHON" -u tools/bench_multi_lora.py --model-dir "$MODEL_DIR" --fleet-dir "$FLEET_DIR" \
  --compressed-dir "$COMPRESSED_DIR" --image "$MODEL_DIR/diffusers_config/examples/i2v_input.JPG" \
  --prompts configs/remade_pilot_prompts.json --output "artifacts/multi_lora/$SLURM_JOB_ID/bench.json"
