#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=rq3-LRcount
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_lorare_counts_%j.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
python - <<'PY'
from collections import Counter
from datasets import load_dataset
d=load_dataset('Styxxxx/LoraRetriever_EvalSet',split='test')
c=Counter(d['task'])
print('rows',len(d),'tasks',len(c),'min',min(c.values()),'median',sorted(c.values())[len(c)//2],'max',max(c.values()))
print(sorted(c.items(), key=lambda x:(x[1],x[0])))
PY
