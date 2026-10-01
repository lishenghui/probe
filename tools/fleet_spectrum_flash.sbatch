#!/usr/bin/env bash
# Submit from repository root; pass live account, partition, GRES and log path on CLI.
#SBATCH --job-name=probe-fleet-spectrum
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -euo pipefail
source "$HOME/.config/hpc-agent/bootstrap.sh"
: "${PROBE_PYTHON:?Set the verified repository interpreter}"
: "${FLEET_DIR:?Set the fleet directory containing manifest.json}"
: "${PROBE_CUDA_HOME:?Set a CUDA toolkit with nvcc for the FlashTSQR JIT build}"
# JIT build needs the interpreter's ninja and an nvcc matching torch's CUDA.
export CUDA_HOME="$PROBE_CUDA_HOME"
export PATH="$(dirname "$PROBE_PYTHON"):$CUDA_HOME/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
SCR=${PROBE_SCRATCH:-/nobackup/proj/disk/bloom/personal/shenghui/scratch}
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$SCR/torch_extensions"
export TMPDIR="$SCR/probe-fleet-spectrum-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
nvcc --version | tail -1; ninja --version
"$PROBE_PYTHON" -u tools/fleet_spectrum_flash.py --fleet-dir "$FLEET_DIR" \
  --output-dir "artifacts/fleet_spectrum/$SLURM_JOB_ID"
