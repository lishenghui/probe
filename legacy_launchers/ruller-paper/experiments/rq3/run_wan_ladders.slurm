#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --job-name=wanladder
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_ladders_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1
D=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache
# The ladders are intermediate training checkpoints of four adapters. Sec. 4.3's
# lambda-intervention varies S by rescaling a released adapter, which a reviewer can
# call artificial; these vary S along a real training trajectory at fixed task,
# fixed data and fixed rank, so they are the natural-experiment version of it.
python -u - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
DEST = Path("/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/wan-ladders")
lad = json.loads(Path("artifacts/rq3/results/wan_ladders.json").read_text())
out = {}
for repo, rs in lad.items():
    for r in rs:
        name = f"{repo.split('/')[-1][:24]}__{Path(r['file']).stem[-12:]}"
        d = DEST / name; d.mkdir(parents=True, exist_ok=True)
        f = hf_hub_download(repo, r["file"], local_dir=d)
        out[name] = {"repo": repo, "path": f, "S": r["S"], "rank": r["rank"],
                     "owner": repo, "ladder": repo}
        print(f"  {name[:44]:46s} S={r['S']:.5f}", flush=True)
(DEST / "ladders.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"{len(out)} checkpoints across {len(lad)} ladders")
PY
python -u ruller-paper/experiments/rq3/wan_output_divergence.py \
  --base $D/wan21-t2v-1.3b \
  --pool $D/wan-ladders/ladders.json \
  --output artifacts/rq3/results/wan_ladders_dv.json
