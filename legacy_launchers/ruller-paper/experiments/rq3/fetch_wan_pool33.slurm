#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --job-name=wan-fetch33
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_fetch33_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1
python -u - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
DEST = Path("/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/wan-loras")
pool = json.loads(Path("artifacts/rq3/results/wan_pool_final.json").read_text())
have = json.loads((DEST / "pool.json").read_text())
out, fail = dict(have), []
for r in pool:
    name = r["name"]
    if name in out:                      # panowan/ultrawan1k/ultrawan4k/longcat
        out[name]["S"] = r["S"]; continue
    try:
        d = DEST / name
        d.mkdir(parents=True, exist_ok=True)
        f = hf_hub_download(r["repo"], r["file"], local_dir=d)
        out[name] = {"repo": r["repo"], "path": f, "S": r["S"],
                     "rank": r["rank"], "owner": r["repo"].split("/")[0]}
        print(f"  {name[:44]:46s} S={r['S']:.5f} {Path(f).stat().st_size/1e6:7.1f} MB", flush=True)
    except Exception as exc:
        fail.append((name, f"{type(exc).__name__} {exc}"[:90]))
        print(f"  FAIL {name}: {type(exc).__name__}", flush=True)
for n, e in out.items():
    e.setdefault("owner", "__have__")
(DEST / "pool33.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"\n{len(out)} adapters in pool33.json, {len(fail)} failures")
for n, e in fail: print("  FAILED", n, e)
PY
du -sh /nobackup/proj/disk/bloom/personal/shenghui/hf_cache/wan-loras
