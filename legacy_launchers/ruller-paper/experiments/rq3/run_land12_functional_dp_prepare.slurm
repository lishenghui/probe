#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=rq3-L12-fDPprep
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_land12_functional_dp_prepare_%j.out
set -euo pipefail
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
python3 ruller-paper/experiments/rq3/solve_two_level_fleet.py \
  --curves 'artifacts/rq3/results/functional_dp_output_land12_*.json' --budget 1321 \
  --output artifacts/rq3/results/functional_dp_land12_b1321.json
python3 ruller-paper/experiments/rq3/solve_two_level_fleet.py \
  --curves 'artifacts/rq3/results/functional_dp_output_land12_*.json' --budget 2289 \
  --output artifacts/rq3/results/functional_dp_land12_b2289.json
