#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=ad_stack_spectrum
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/animatediff_stack_spectrum_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora

WORKSPACE=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
RUNROOT=/tmp/${USER}/animatediff_stack_spectrum_${SLURM_JOB_ID}
INPUT=$RUNROOT/MotionLoRA
PYDEPS=$RUNROOT/python
OUTPUT=$WORKSPACE/artifacts/animatediff_fraq_stack/spectrum_weighted_4lora
mkdir -p "$INPUT" "$PYDEPS" "$OUTPUT"

python -m pip install --target "$PYDEPS" matplotlib
export PYTHONPATH="$PYDEPS:${PYTHONPATH:-}"

for name in ZoomIn PanRight TiltUp RollingClockwise; do
  hf download guoyww/animatediff "v2_lora_${name}.ckpt" --local-dir "$INPUT"
done

cd "$WORKSPACE"
python -u tools/analyze_weighted_lora_stack_spectrum.py \
  --checkpoint "$INPUT/v2_lora_ZoomIn.ckpt" --weight 0.60 \
  --checkpoint "$INPUT/v2_lora_PanRight.ckpt" --weight 0.45 \
  --checkpoint "$INPUT/v2_lora_TiltUp.ckpt" --weight 0.35 \
  --checkpoint "$INPUT/v2_lora_RollingClockwise.ckpt" --weight 0.25 \
  --output-dir "$OUTPUT" --device cuda
