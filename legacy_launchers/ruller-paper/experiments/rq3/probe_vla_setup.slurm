#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --job-name=rq3-vlaprobe
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_vlaprobe_%j.out
set -euo pipefail
# Times the RLinf/LIBERO install and a minimal eval, to decide whether
# same-observation probing on the VLA policy is affordable before the deadline.
START=$SECONDS
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
R_SRC="$W/artifacts/rq3/vendor/RLinf"
R="/tmp/${USER}/RLinf_${SLURM_JOB_ID}"
VENV="/tmp/${USER}/rlinf_${SLURM_JOB_ID}"
mkdir -p "/tmp/${USER}"
cp -a "$R_SRC" "$R"
echo "[t=$((SECONDS-START))s] copied RLinf"
export LIBERO_CONFIG_PATH="/tmp/${USER}/libero_config_${SLURM_JOB_ID}"
mkdir -p "$LIBERO_CONFIG_PATH"
export REPO_PATH="$R" EMBODIED_PATH="$R/examples/embodiment"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME="/tmp/${USER}/hf_${SLURM_JOB_ID}" TOKENIZERS_PARALLELISM=false
cd "$R"
bash requirements/install.sh embodied --model openvla-oft --env libero \
  --venv "$VENV" --no-root --no-flash-attn --no-apex --install-rlinf
echo "[t=$((SECONDS-START))s] install finished"
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1
python - <<'PY'
import time
t0=time.time()
import torch, importlib
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
for m in ("libero","transformers","peft"):
    try:
        mod=importlib.import_module(m); print(" ",m,"ok",getattr(mod,"__version__","?"))
    except Exception as e: print(" ",m,"FAIL",type(e).__name__,str(e)[:80])
print(f"import probe {time.time()-t0:.1f}s")
PY
echo "[t=$((SECONDS-START))s] TOTAL SETUP"
