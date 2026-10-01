#!/usr/bin/env bash
# Submit from repository root; pass live account, partition, GRES and log path on CLI.
# Env: PROBE_SCRATCH (optional), PROBE_PYTHON, PROBE_CUDA_HOME, FLEET_DIR, MODEL_DIR, SPECTRUM_RUN; optional STEPS, FRAMES, ADAPTERS, BUDGETS.
#SBATCH --job-name=probe-wan-pilot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
set -euo pipefail
source "$HOME/.config/hpc-agent/bootstrap.sh"
: "${PROBE_PYTHON:?}" "${PROBE_CUDA_HOME:?}" "${FLEET_DIR:?}" "${MODEL_DIR:?}" "${SPECTRUM_RUN:?}"
cd "$SLURM_SUBMIT_DIR"
SCR=${PROBE_SCRATCH:-/nobackup/proj/disk/bloom/personal/shenghui/scratch}
export PYTHONDONTWRITEBYTECODE=1 CUDA_HOME="$PROBE_CUDA_HOME"
export PATH="$(dirname "$PROBE_PYTHON"):$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SCR/torch_extensions" PIP_CACHE_DIR="$SCR/pip_cache"
export TMPDIR="$SCR/probe-wan-pilot-$SLURM_JOB_ID" HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR"
OUT=artifacts/wan_fleet_pilot/$SLURM_JOB_ID
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PROBE_PYTHON" -c "import imageio, imageio_ffmpeg, ftfy" 2>/dev/null || \
  "$PROBE_PYTHON" -m pip install --quiet imageio imageio-ffmpeg ftfy
COMP="$FLEET_DIR/compressed/$SLURM_JOB_ID"
"$PROBE_PYTHON" -u tools/compress_fleet_flash.py --fleet-dir "$FLEET_DIR" --spectrum-run "$SPECTRUM_RUN" \
  --adapters ${ADAPTERS:-Assassin Rotate angry-face} --budgets ${BUDGETS:-0.75 0.5 0.25} --output-dir "$COMP"
mkdir -p "$OUT"; cp "$COMP/summary.json" "$OUT/compression_summary.json"
"$PROBE_PYTHON" - "$FLEET_DIR" "$COMP" "$OUT/plan.json" ${ADAPTERS:-Assassin Rotate angry-face} <<'PY'
import json, sys
from pathlib import Path
fleet, comp, out, names = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
prompts = json.loads(Path('configs/remade_pilot_prompts.json').read_text())
entries = {e['name']: e for e in json.loads((fleet / 'manifest.json').read_text())['entries']}
plan = []
for n in names:
    variants = {'original': str(fleet / entries[n]['path'])}
    if n == names[0]:  # determinism control: same adapter, same seed, should deviate ~0
        variants['original_repeat'] = variants['original']
    for b in sorted(comp.glob('b*'), key=lambda p: -int(p.name[1:])):
        variants[b.name] = str(next((b / n).glob('*.safetensors')))
    plan.append(dict(adapter=n, prompt=prompts[n], variants=variants))
out.write_text(json.dumps(plan, indent=1))
PY
"$PROBE_PYTHON" -u tools/generate_wan_i2v_fleet_pilot.py --model-dir "$MODEL_DIR" --plan "$OUT/plan.json" \
  --image "$MODEL_DIR/diffusers_config/examples/i2v_input.JPG" --output-dir "$OUT/videos" \
  --steps "${STEPS:-30}" --frames "${FRAMES:-49}"
