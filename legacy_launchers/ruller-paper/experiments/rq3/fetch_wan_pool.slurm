#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=wan-fetch
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/wan_fetch_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
W=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge; cd "$W"
# Models go to the shared cache: /tmp is node-local and every later shard needs these.
# HF_HOME stays on /tmp because Lustre reports the project quota as the filesystem
# size, so datasets' free-space check sees zero and refuses to write.
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1
DEST=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache
python -u - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path
DEST = Path("/nobackup/proj/disk/bloom/personal/shenghui/hf_cache")

print("=== base: Wan2.1-T2V-1.3B (diffusers) ===", flush=True)
p = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", local_dir=DEST / "wan21-t2v-1.3b")
print("  ->", p, flush=True)

print("=== CLIP for the text-video alignment probe ===", flush=True)
p = snapshot_download("openai/clip-vit-large-patch14", local_dir=DEST / "clip-vit-large-p14")
print("  ->", p, flush=True)

# solace from the original screen is unrecoverable: it appears in no HF search and
# nowhere on disk, and the spec file that drove that run was never saved. Its
# S = 0.0091 sat mid-range, so dropping it leaves the spread at 8.7x.
ADAPTERS = [
    ("panowan",    "YOUSIKI/PanoWan",                            "latest-lora.ckpt"),
    ("ultrawan1k", "APRIL-AIGC/UltraWan",                        "ultrawan-1k.ckpt"),
    ("ultrawan4k", "APRIL-AIGC/UltraWan",                        "ultrawan-4k.ckpt"),
    ("longcat",    "lightx2v/Wan2.1-T2V-1.3B-longcat-step1500",  "adapter_model.safetensors"),
]
out = {}
for name, repo, fname in ADAPTERS:
    d = DEST / "wan-loras" / name
    d.mkdir(parents=True, exist_ok=True)
    f = hf_hub_download(repo, fname, local_dir=d)
    out[name] = {"repo": repo, "path": f}
    print(f"  {name:11s} {Path(f).stat().st_size/1e6:8.1f} MB  {f}", flush=True)

import json
spec = DEST / "wan-loras" / "pool.json"
spec.write_text(json.dumps(out, indent=2) + "\n")
print("wrote", spec)
PY
du -sh "$DEST/wan21-t2v-1.3b" "$DEST/wan-loras" "$DEST/clip-vit-large-p14"
