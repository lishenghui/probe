#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J smolvla_lora2
#SBATCH -o /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/smolvla_lora2_%j.log
#SBATCH -e /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/smolvla_lora2_%j.err

set -uo pipefail
LEROBOT=/nobackup/proj/disk/bloom/personal/shenghui/venvs/lerobot
SMOLVLA=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/third_party/CORAL/smolvla
export PATH="$LEROBOT/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME="$SMOLVLA/hf_home"
export HF_LEROBOT_HOME="$SMOLVLA/lerobot_home"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=0
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

BASE="$SMOLVLA/pretrained/smolvla_libero"
DS="$SMOLVLA/datasets/libero"
STEPS=${STEPS:-2000}
BATCH=${BATCH:-32}

echo "arch=$(uname -m) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
"$LEROBOT/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

train_one () {
  local name="$1" episodes="$2"
  local out="$SMOLVLA/outputs/${name}_r16"
  echo ""
  echo "############################################################"
  echo "### SmolVLA LoRA r16: ${name}  (steps=$STEPS batch=$BATCH)"
  echo "############################################################"
  lerobot-train \
    --policy.path="$BASE" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.input_features=null \
    --policy.output_features=null \
    --policy.optimizer_lr=1e-3 \
    --policy.scheduler_decay_lr=1e-4 \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root="$DS" \
    --dataset.episodes="$episodes" \
    --dataset.video_backend=pyav \
    --steps="$STEPS" \
    --batch_size="$BATCH" \
    --save_freq=1000 \
    --num_workers=4 \
    --peft.method_type=LORA \
    --peft.r=16 \
    --peft.lora_alpha=32 \
    --output_dir="$out" \
    --job_name="smolvla_${name}_r16"
}

SPATIAL_T2="[1283,1287,1289,1299,1310,1315,1316,1317,1320,1345,1353,1354,1358,1377,1380,1383,1385,1426,1428,1432,1440,1442,1445,1455,1460,1466,1478,1479,1483,1509,1510,1514,1545,1549,1556,1557,1573,1603,1622,1641,1652,1657,1665,1668,1682,1686]"
GOAL_T8="[379,422,426,431,433,447,448,451,459,466,481,483,488,507,511,513,522,532,537,549,551,563,568,582,607,615,620,621,626,634,639,642,646,653,655,670,679,708,716,718,726,727,749,750,768,770,801,803,806]"

train_one "spatial_t2" "$SPATIAL_T2"
train_one "goal_t8" "$GOAL_T8"

echo ""
echo "=== DONE ==="
find "$SMOLVLA/outputs" -name "adapter_model.safetensors" -exec ls -la {} \;
