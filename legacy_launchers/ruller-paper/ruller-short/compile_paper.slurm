#!/bin/bash
# The login node is x86_64 and the compute nodes are aarch64, so the tectonic
# build has to match wherever this runs.  Picking the wrong one fails with
# "cannot execute binary file", which is easy to mistake for a successful build
# because the previous main.pdf is left in place -- hence the rm below.
# LaTeX needs no GPU and no queue, so prefer running this on the login node:
#
#   bash compile_paper.slurm
#
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=ruller-tex
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/ruller_tex_%j.out
set -euo pipefail

ENVS=/nobackup/proj/disk/bloom/personal/shenghui/conda_envs
case "$(uname -m)" in
  x86_64)  TECTONIC=$ENVS/tex_x86/bin/tectonic ;;
  aarch64) TECTONIC=$ENVS/hfedlora/bin/tectonic ;;
  *) echo "no tectonic for $(uname -m)" >&2; exit 1 ;;
esac

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
rm -f main.pdf
"$TECTONIC" main.tex
python3 - <<'COUNT'
import re, zlib
d = open('main.pdf', 'rb').read()
n = 0
for m in re.finditer(rb'stream\r?\n', d):
    a = m.end()
    b = d.find(b'endstream', a)
    try:
        n += len(re.findall(rb'/Type\s*/Page[^s]', zlib.decompress(d[a:b])))
    except Exception:
        pass
print('PAGES:', n)
COUNT
