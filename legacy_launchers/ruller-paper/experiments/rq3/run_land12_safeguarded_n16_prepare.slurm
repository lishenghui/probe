#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=rq3-L12-safe16prep
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_land12_safe16_prepare_%j.out
set -euo pipefail
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
python3 ruller-paper/experiments/rq3/build_safeguarded_curves.py \
  --candidate gamma2 'artifacts/rq3/results/sensitivity_land12_n16/*.json' 2 \
  --candidate spectral 'artifacts/rq3/results/spectral_output_land12_n16_*.json' 0 \
  --output-dir artifacts/rq3/results/safeguarded_land12_n16 \
  --prefix safeguarded_land12_n16
python3 ruller-paper/experiments/rq3/solve_two_level_fleet.py \
  --curves 'artifacts/rq3/results/safeguarded_land12_n16/*.json' --budget 1321 \
  --output artifacts/rq3/results/safeguarded_land12_n16_b1321.json
