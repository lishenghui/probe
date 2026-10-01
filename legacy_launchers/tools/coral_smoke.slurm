#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -J coral_smoke
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/coral_smoke_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/coral_smoke_%j.err

set -euo pipefail
CORAL=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/CORAL
VENV=/nobackup/proj/disk/bloom/personal/shenghui/venvs/simvla
export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$CORAL"
echo "arch=$(uname -m) host=$(hostname)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

accelerate launch --num_processes=1 --mixed_precision bf16 \
  train_coral_smolvlm.py \
  --base_model ./pretrained/SimVLA-LIBERO \
  --lora_name smoke \
  --lora_rank 16 \
  --lora_alpha 32 \
  --train_metas_path ./datasets/coral_metas/libero_spatial_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.json \
  --output_dir ./lora_adapters/smoke \
  --learning_rate 5e-5 \
  --iters 3 \
  --warmup_steps 1 \
  --batch_size 2 \
  --action_mode libero_joint \
  --norm_stats_path ./norm_stats/libero_norm.json \
  --save_interval 3 \
  --log_interval 1 \
  --num_workers 0 \
  --num_actions 10 \
  --image_size 384

echo "=== SMOKE DONE ==="
ls -la ./lora_adapters/smoke/ 2>/dev/null
