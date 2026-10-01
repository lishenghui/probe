#!/usr/bin/env bash
# Compress the whole fleet with FlashTSQR, then measure end-to-end residency. Submit from repo root.
# Env: PROBE_PYTHON, PROBE_CUDA_HOME, FLEET_DIR, MODEL_DIR, SPECTRUM_RUN; optional PROBE_SCRATCH,
#      BUDGETS, ATTN, FRAMES, STEPS, COMPRESSED_DIR (reuse an earlier full-fleet compression).
#SBATCH --job-name=probe-fleet-residency
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
set -euo pipefail
source "$HOME/.config/hpc-agent/bootstrap.sh"
: "${PROBE_PYTHON:?}" "${PROBE_CUDA_HOME:?}" "${FLEET_DIR:?}" "${MODEL_DIR:?}" "${SPECTRUM_RUN:?}"
cd "$SLURM_SUBMIT_DIR"
SCR=${PROBE_SCRATCH:-/nobackup/proj/disk/bloom/personal/shenghui/scratch}
export PYTHONDONTWRITEBYTECODE=1 CUDA_HOME="$PROBE_CUDA_HOME" HF_HUB_OFFLINE=1
export PATH="$(dirname "$PROBE_PYTHON"):$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SCR/torch_extensions" TMPDIR="$SCR/probe-fleet-residency-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
OUT=artifacts/fleet_residency/$SLURM_JOB_ID; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
COMP=${COMPRESSED_DIR:-$SCR/fleet_compressed/$SLURM_JOB_ID}
if [ -z "${COMPRESSED_DIR:-}" ]; then
  ALL=$("$PROBE_PYTHON" -c "import json,sys;print(' '.join(e['name'] for e in json.load(open(sys.argv[1]))['entries']))" "$FLEET_DIR/manifest.json")
  "$PROBE_PYTHON" -u tools/compress_fleet_flash.py --fleet-dir "$FLEET_DIR" --spectrum-run "$SPECTRUM_RUN" \
    --adapters $ALL --budgets ${BUDGETS:-0.75 0.5 0.25} --output-dir "$COMP"
fi
cp "$COMP/summary.json" "$OUT/compression_summary.json"
"$PROBE_PYTHON" -u tools/bench_fleet_residency.py --model-dir "$MODEL_DIR" --fleet-dir "$FLEET_DIR" \
  --compressed-dir "$COMP" --image "$MODEL_DIR/diffusers_config/examples/i2v_input.JPG" \
  --prompts configs/remade_pilot_prompts.json --output "$OUT/residency.json" \
  --attention-backend "${ATTN:-native}" --frames "${FRAMES:-49}" --steps "${STEPS:-30}"
