#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --job-name=rq3-alloc3
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_alloc3_%j.out
set -euo pipefail
# Does a smarter rank allocator rescue the VLA at a fixed parameter budget?
# waterfill (min sum P_l^2) already lost to uniform energy truncation, so this
# separates the allocator's form from the fact that it zeroes whole modules:
#   ALL    - every direction kept; a rebuild sanity check, must score D_a ~ 0
#   TAU    - equalise max_l P_l, the S_l*sqrt(1-tau_l) <= eps form
#   WF4    - waterfill with a rank-4 floor, budget-corrected
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; A=$W/artifacts/rq3/adapters
cd "$W"
B="--reference $A/openvla130-corrections/e90 --base artifacts/rq3/models/openvla-libero130-base"
for T in e95 e90; do
  python -u ruller-paper/experiments/rq3/build_strength_allocated_adapter.py $B \
    --budget-from $A/openvla130-corrections/$T --copy-ranks \
    --output $A/openvla130-alloc/uniform-rebuilt-$T
done

R="/tmp/${USER}/RLinf_${SLURM_JOB_ID}"; VENV="/tmp/${USER}/rlinf_${SLURM_JOB_ID}"
mkdir -p "/tmp/${USER}"; cp -a "$W/artifacts/rq3/vendor/RLinf" "$R"
export LIBERO_CONFIG_PATH="/tmp/${USER}/libero_config_${SLURM_JOB_ID}"; mkdir -p "$LIBERO_CONFIG_PATH"
export REPO_PATH="$R" EMBODIED_PATH="$R/examples/embodiment"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME="/tmp/${USER}/hf_${SLURM_JOB_ID}" TOKENIZERS_PARALLELISM=false
cd "$R"
bash requirements/install.sh embodied --model openvla-oft --env libero \
  --venv "$VENV" --no-root --no-flash-attn --no-apex --install-rlinf
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONPATH="$R:${PYTHONPATH:-}"
cd "$W"
python -u ruller-paper/experiments/rq3/replay_vla_observations.py \
  --model artifacts/rq3/models/openvla-libero130-base \
  --eval-config "$R/evaluations/libero/libero_130_openvlaoft_eval.yaml" \
  --observations artifacts/rq3/results/openvla130/observations \
  --variant $A/openvla130-alloc/uniform-rebuilt-e95 $A/openvla130-alloc/uniform-rebuilt-e90 \
  --output artifacts/rq3/results/local_divergence_vla_alloc3.json
