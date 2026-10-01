#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -t 01:00:00
#SBATCH -J coral_r16_preview
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/coral_r16_preview_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/coral_r16_preview_%j.err

set -euo pipefail
CORAL=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/CORAL
VENV=/nobackup/proj/disk/bloom/personal/shenghui/venvs/simvla
export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$CORAL"
echo "arch=$(uname -m) host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ndev', torch.cuda.device_count())"

TASKS=(
  "libero_spatial|pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
  "libero_goal|put_the_bowl_on_the_plate"
)

for t in "${TASKS[@]}"; do
  SUITE="${t%%|*}"
  SLUG="${t##*|}"
  META="./datasets/coral_metas/${SUITE}_${SLUG}.json"
  OUT="./lora_adapters/coral_r16_preview/${SUITE}/${SLUG}"
  mkdir -p "$OUT"
  echo ""
  echo "############################################################"
  echo "### Training: ${SUITE} / ${SLUG}"
  echo "############################################################"

  accelerate launch --num_processes=4 --mixed_precision bf16 \
    train_coral_smolvlm.py \
    --base_model ./pretrained/SimVLA-LIBERO \
    --lora_name "${SUITE}_${SLUG}" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --train_metas_path "$META" \
    --output_dir "$OUT" \
    --learning_rate 5e-5 \
    --iters 50 \
    --warmup_steps 10 \
    --batch_size 64 \
    --action_mode libero_joint \
    --norm_stats_path ./norm_stats/libero_norm.json \
    --save_interval 50 \
    --log_interval 10 \
    --num_workers 4 \
    --num_actions 10 \
    --image_size 384 \
    2>&1 | tee "$OUT/train.log"
done

echo ""
echo "=== PREVIEW DONE ==="
find ./lora_adapters/coral_r16_preview -name "adapter_model.safetensors" -exec ls -la {} \;
