#!/bin/bash
#SBATCH -A naiss2025-22-1535-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --array=0-77
#SBATCH --job-name=rq3-fdp0probe
#SBATCH --output=/nobackup/proj/disk/bloom/personal/shenghui/LoRAForge/logs/rq3_fdp0probe_%A_%a.out
set -euo pipefail
module load GPU/Miniforge/26.3.2-2-eb
mamba activate /nobackup/proj/disk/bloom/personal/shenghui/conda_envs/hfedlora
cd /nobackup/proj/disk/bloom/personal/shenghui/LoRAForge
export HF_HOME=/tmp/${USER}/hf_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" logs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
I=$SLURM_ARRAY_TASK_ID
if [ "$I" -lt 12 ]; then
  TASKS=(conllpp dbpedia e2e_nlg glue_cola glue_mrpc glue_qnli glue_qqp glue_sst2 gsm8k hellaswag viggo wikisql)
  T=${TASKS[$I]}; POOL=land
  BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
  OLD="artifacts/rq3/results/layer_probe_${T}.json"; OUT="artifacts/rq3/results/layer_probe0_${T}.json"
  EXTRA=(--example-start 200)
elif [ "$I" -lt 37 ]; then
  TASKS=(task0112 task0149 task0223 task0260 task0297 task0334 task0371 task0408 task0593 task0852 task0889 task0926 task1037 task1111 task1370 task1407 task1444 task1518 task1592 task0704 task0778 task0815 task0963 task1148 task1333)
  T=${TASKS[$((I-12))]}; POOL=cts
  BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a
  OLD="artifacts/rq3/results/layer_probe_cts25_${T}.json"; OUT="artifacts/rq3/results/layer_probe0_cts25_${T}.json"
  EXTRA=(--example-start 0 --new 32)
else
  TASKS=(anli_r1_10templates anli_r2_10templates anli_r3_10templates arc_challenge_10templates bool_q_10templates common_gen_10templates copa_10templates cosmos_qa_10templates dart_10templates drop_10templates e2e_nlg_10templates glue_mrpc_10templates glue_qqp_10templates hellaswag_10templates imdb_reviews_10templates mnli_mismatched_10templates multirc_10templates natural_questions_10templates openbookqa_10templates para_crawl_enes_10templates paws_wiki_10templates piqa_10templates record_10templates sentiment140_10templates snli_10templates squad_v1_10templates squad_v2_10templates sst2_10templates story_cloze_10templates stsb_10templates trivia_qa_10templates web_nlg_en_10templates wmt14_enfr_10templates wmt16_translate_csen_10templates wmt16_translate_deen_10templates wmt16_translate_fien_10templates wmt16_translate_roen_10templates wmt16_translate_ruen_10templates wmt16_translate_tren_10templates wnli_10templates yelp_polarity_reviews_10templates)
  T=${TASKS[$((I-37))]}; POOL=lorare
  BASE=/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf
  OLD="artifacts/rq3/results/layer_probe_lorare_${T}.json"; OUT="artifacts/rq3/results/layer_probe0_lorare_${T}.json"
  EXTRA=(--example-start 0 --new 48)
fi
python -u ruller-paper/experiments/rq3/layer_truncation_probe.py \
  --base "$BASE" --pool "$POOL" --task "$T" \
  --work /tmp/${USER}/fdp0probe_${SLURM_JOB_ID} \
  --examples 16 --batch 4 "${EXTRA[@]}" --single-ranks 0 \
  --random-allocations 0 --merge-existing "$OLD" --output "$OUT"
