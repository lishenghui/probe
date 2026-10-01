#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --job-name=wan_smoke
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_smoke_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
NV=$W/nvcc_env
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
REPO=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/FlashTSQR
cd "$REPO/bench"
export CUDA_HOME="$NV" PATH="$NV/bin:$PATH" TORCH_CUDA_ARCH_LIST="9.0" TORCH_EXTENSIONS_DIR=$W/.torch_ext
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name --format=csv,noheader
echo "===== SMOKE: exec side (4 layers, 17f) ====="
python -u bench_wan_multilora.py --layers 4 --frames 17 --reps 3 || echo "EXEC FAILED rc=$?"
echo "===== SMOKE: merge side ====="
python -u bench_wan_merge.py --k 32 --reps 3 --no-loop || echo "MERGE FAILED rc=$?"
