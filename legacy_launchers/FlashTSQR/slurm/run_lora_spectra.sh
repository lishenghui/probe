#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=wan_lora_spectra
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_lora_spectra_%j.out
set -euo pipefail

module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora

W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
NV=$W/nvcc_env
INPUT=$W/artifacts/lightx2v/Wan2.2-Distill-Loras
OUTPUT=$INPUT/fraq_spectrum_flash
cd "$W"

if [ ! -x "$NV/bin/nvcc" ]; then
    echo "Creating CUDA 12.6 compiler environment"
    mamba create -y -q -p "$NV" -c nvidia cuda-nvcc=12.6 cuda-cudart-dev=12.6
fi
test -x "$NV/bin/nvcc"
export CUDA_HOME="$NV"
export PATH="$NV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=$W/.torch_ext

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import torch
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
PY

for checkpoint in "$INPUT"/*.safetensors; do
    name=$(basename "$checkpoint" .safetensors)
    output_dir="$OUTPUT/$name"
    mkdir -p "$output_dir"
    echo "===== $name ====="
    python -u tools/analyze_lora_fraq_spectrum.py \
        "$checkpoint" --output-dir "$output_dir" --device cuda --backend flash \
        > "$output_dir/run_cuda.log"
    tail -25 "$output_dir/run_cuda.log"
done

echo "All spectra complete."
