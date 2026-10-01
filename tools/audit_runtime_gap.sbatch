#!/usr/bin/env bash
# Submit from repository root; pass live account, partition, GRES and log paths on CLI.
#SBATCH --job-name=probe-runtime-audit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
set -euo pipefail
source "$HOME/.config/hpc-agent/bootstrap.sh"
: "${PROBE_PYTHON:?Set the verified repository interpreter}"
: "${PROBE_SOURCE:?Set the read-only historical source checkout}"
: "${SLURM_SUBMIT_DIR:?Submit through Slurm}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export TMPDIR="$SCRATCH_ROOT/probe-runtime-audit-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "Starting factor-memory and historical evidence audit; job=$SLURM_JOB_ID"
"$PROBE_PYTHON" -u tools/audit_runtime_gap.py \
  --source "$PROBE_SOURCE" --results baselines/legacy/results \
  --gpu-memory --output "artifacts/runtime_audit/$SLURM_JOB_ID/audit.json"
