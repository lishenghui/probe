#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=wan_lora
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_lora_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora

REPO=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/FlashTSQR
cd "$REPO/bench"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================================="
echo "== Wan 2.1 1.3B, 49 frames @ 480p (20280 tokens)        =="
echo "=========================================================="
python -u bench_wan_multilora.py --frames 49 --height 480 --width 832 \
       --steps 50 --json "$REPO/bench/out_wan_49f.json"

echo
echo "=========================================================="
echo "== Wan 2.1 1.3B, 81 frames @ 480p (32760 tokens)        =="
echo "=========================================================="
python -u bench_wan_multilora.py --frames 81 --height 480 --width 832 \
       --steps 50 --no-kernels --json "$REPO/bench/out_wan_81f.json"

echo
echo "=========================================================="
echo "== short clip: 17 frames @ 480p (7800 tokens)           =="
echo "=========================================================="
python -u bench_wan_multilora.py --frames 17 --height 480 --width 832 \
       --steps 50 --no-kernels --json "$REPO/bench/out_wan_17f.json"
