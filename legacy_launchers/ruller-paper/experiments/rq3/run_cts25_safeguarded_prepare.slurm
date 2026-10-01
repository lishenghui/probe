#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=rq3-CTS25-safeprep
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_cts25_safeguarded_prepare_%j.out
set -euo pipefail
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
python3 ruller-paper/experiments/rq3/build_safeguarded_curves.py \
  --candidate gamma2 'artifacts/rq3/results/squared_output_cts25_task*.json' 2 \
  --candidate spectral 'artifacts/rq3/results/spectral_output_cts25_*.json' 0 \
  --output-dir artifacts/rq3/results/safeguarded_cts25 \
  --prefix safeguarded_cts25
python3 ruller-paper/experiments/rq3/solve_two_level_fleet.py \
  --curves 'artifacts/rq3/results/safeguarded_cts25/*.json' --budget 8811 \
  --output artifacts/rq3/results/safeguarded_cts25_b8811.json
python3 ruller-paper/experiments/rq3/solve_two_level_fleet.py \
  --curves 'artifacts/rq3/results/safeguarded_cts25/*.json' --budget 19202 \
  --output artifacts/rq3/results/safeguarded_cts25_b19202.json
