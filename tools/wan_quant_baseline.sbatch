#!/usr/bin/env bash
# LoRA precision baselines vs rank compression at matched bytes, one adapter per job. Submit from repo root.
# Env: PROBE_PYTHON, FLEET_DIR, MODEL_DIR, COMPRESSED_DIR (FlashTSQR b50/b25 outputs), ADAPTER; optional PROBE_SCRATCH.
#SBATCH --job-name=probe-wan-quant
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:15:00
set -euo pipefail
source "$HOME/.config/hpc-agent/bootstrap.sh"
: "${PROBE_PYTHON:?}" "${FLEET_DIR:?}" "${MODEL_DIR:?}" "${COMPRESSED_DIR:?}" "${ADAPTER:?}"
cd "$SLURM_SUBMIT_DIR"
SCR=${PROBE_SCRATCH:-/nobackup/proj/disk/bloom/personal/shenghui/scratch}
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TMPDIR="$SCR/probe-wan-quant-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
OUT=artifacts/wan_quant_baseline/$SLURM_JOB_ID; mkdir -p "$OUT"
"$PROBE_PYTHON" - "$FLEET_DIR" "$COMPRESSED_DIR" "$ADAPTER" "$OUT/plan.json" <<'PY'
import json, sys
from pathlib import Path
fleet, comp, name, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
entry = {e['name']: e for e in json.loads((fleet / 'manifest.json').read_text())['entries']}[name]
orig = str(fleet / entry['path'])
rank = {b: str(next((comp / b / name).glob('*.safetensors'))) for b in ('b50', 'b25')}
variants = {  # ~100% | ~50% | ~25% | ~13% of the original bytes
    'original': orig,
    'fp8': dict(path=orig, quant='fp8'), 'int8': dict(path=orig, quant='int8'), 'b50': rank['b50'],
    'int4': dict(path=orig, quant='int4'), 'b25': rank['b25'],
    'b50+fp8': dict(path=rank['b50'], quant='fp8'), 'b50+int8': dict(path=rank['b50'], quant='int8'),
    'b25+fp8': dict(path=rank['b25'], quant='fp8'),
}
prompt = json.loads(Path('configs/remade_pilot_prompts.json').read_text())[name]
out.write_text(json.dumps([dict(adapter=name, prompt=prompt, variants=variants)], indent=1))
PY
"$PROBE_PYTHON" -u tools/generate_wan_i2v_fleet_pilot.py --model-dir "$MODEL_DIR" --plan "$OUT/plan.json" \
  --image "$MODEL_DIR/diffusers_config/examples/i2v_input.JPG" --output-dir "$OUT/videos"
