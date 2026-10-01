#!/bin/bash
# Regenerate the Lots-of-LoRAs allocation artifacts.
#
# Both commands below were recovered by bisecting against the stored artifacts
# and reproduce them byte-identically, which is worth writing down because the
# two arms do NOT take the same divergence input and the difference is not
# guessable:
#
#   SCT   fits the two-exponent law on the target pool, so it needs cells across
#         the whole threshold range -- the high-tau sweep alone leaves the fit
#         short at the severe end and moves the allocation at the loosest budget
#         (worst 0.786 instead of 0.906).
#   A-SCT never fits the law, so it takes only the high-tau sweep for pool
#         membership and gets its slope from the two disjoint-prompt anchors.
#
# Pass a suffix to write beside the published artifacts rather than over them.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=artifacts/rq3/results
SUF="${1:-}"

python3 -u ruller-paper/experiments/rq3/grid_budget_allocation.py \
  --task "$R"/cts_task*.json \
  --div "$R"/cts_dout2_shard*.json "$R"/cts_dout_low_shard*.json \
  --nominal 1536 --label "Lots-of-LoRAs-original-SCT" \
  --output "$R/grid_alloc_cts_original_full${SUF}.json"

python3 -u ruller-paper/experiments/rq3/grid_budget_allocation.py \
  --task "$R"/cts_task*.json \
  --div "$R"/cts_dout2_shard*.json \
  --anchor-div "$R"/cts_anchor_train_*.json "$R"/cts_anchor99_train_*.json \
  --nominal 1536 --anchor-tau 0.95 --estimate-b-from-anchors 0.99 0.95 \
  --label "Lots-of-LoRAs-disjoint-two-anchor" \
  --output "$R/grid_alloc_cts_disjoint_anchor2${SUF}.json"
