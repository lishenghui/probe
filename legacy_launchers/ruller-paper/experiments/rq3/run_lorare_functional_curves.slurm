#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --array=0-40
#SBATCH --job-name=rq3-lrfunc
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_lorare_functional_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}; mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf
ALL=(anli_r1_10templates anli_r2_10templates anli_r3_10templates arc_challenge_10templates bool_q_10templates common_gen_10templates copa_10templates cosmos_qa_10templates dart_10templates drop_10templates e2e_nlg_10templates glue_mrpc_10templates glue_qqp_10templates hellaswag_10templates imdb_reviews_10templates mnli_mismatched_10templates multirc_10templates natural_questions_10templates openbookqa_10templates para_crawl_enes_10templates paws_wiki_10templates piqa_10templates record_10templates sentiment140_10templates snli_10templates squad_v1_10templates squad_v2_10templates sst2_10templates story_cloze_10templates stsb_10templates trivia_qa_10templates web_nlg_en_10templates wmt14_enfr_10templates wmt16_translate_csen_10templates wmt16_translate_deen_10templates wmt16_translate_fien_10templates wmt16_translate_roen_10templates wmt16_translate_ruen_10templates wmt16_translate_tren_10templates wnli_10templates yelp_polarity_reviews_10templates)
T=${ALL[$SLURM_ARRAY_TASK_ID]}
echo "task=$T"
python -u ruller-paper/experiments/rq3/land_two_level_curve.py \
  --base "$BASE" --pool lorare --task "$T" --work /tmp/${USER}/lrfunc_${SLURM_JOB_ID} \
  --examples 16 --example-start 0 --batch 4 --new 48 \
  --layer-risk functional --layer-costs "artifacts/rq3/results/layer_probe_lorare_${T}.json" \
  --divergence output --budget-step 8 --max-budget 512 \
  --output "artifacts/rq3/results/functional_dp_output_lorare_${T}.json"
