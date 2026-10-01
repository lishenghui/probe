#!/usr/bin/env bash
# Qwen3-8B Belebele accuracy with/without xummer per-language LoRA. Submit from repo root.
# Env: PROBE_PYTHON, MODEL_DIR, ADAPTER_DIR, DATA_DIR; optional LIMIT, PROBE_SCRATCH.
#SBATCH --job-name=probe-belebele
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
source "$HOME/.config/hpc-agent/bootstrap.sh"
: "${PROBE_PYTHON:?}" "${MODEL_DIR:?}" "${ADAPTER_DIR:?}" "${DATA_DIR:?}"
cd "$SLURM_SUBMIT_DIR"
SCR=${PROBE_SCRATCH:-/nobackup/proj/disk/bloom/personal/shenghui/scratch}
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$SCR/probe-belebele-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
"$PROBE_PYTHON" -u tools/eval_belebele_lora.py --model "$MODEL_DIR" --adapters "$ADAPTER_DIR" --data "$DATA_DIR" \
  ${LIMIT:+--limit $LIMIT} --output "artifacts/belebele_eval/$SLURM_JOB_ID/results.json"
