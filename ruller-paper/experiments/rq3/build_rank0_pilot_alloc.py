#!/usr/bin/env python3
"""Build matched-budget spectral floor-1/floor-0 pilot allocations."""
import json
from pathlib import Path

from two_level_allocation import minimax_allocate


ROOT = Path("artifacts/rq3/results")
NAMES = [
    "anli_r2_10templates", "multirc_10templates", "anli_r1_10templates",
    "story_cloze_10templates", "drop_10templates",
    "yelp_polarity_reviews_10templates", "cosmos_qa_10templates",
    "imdb_reviews_10templates",
]


def curves(prefix):
    return {
        name: json.loads((ROOT / f"{prefix}{name}.json").read_text())["curve"]
        for name in NAMES
    }


base = json.loads((ROOT / "fra2x2_alloc/fra2x2_funcdp_js_b3191.json").read_text())
budget = sum(int(base["allocation"][name]["k"]) for name in NAMES)
outdir = ROOT / "rank0_pilot_alloc"
outdir.mkdir(exist_ok=True)
for label, prefix in (("floor1", "fra2x2_spectral_lorare_"),
                      ("floor0", "rank0_spectral_lorare_")):
    chosen, ceiling, spent = minimax_allocate(curves(prefix), budget)
    doc = {"pool": "LoRARetriever-rank0-pilot", "method": label,
           "currency": "d_js", "n": len(chosen), "budget": budget,
           "spent": spent, "optimal_ceiling": ceiling, "allocation": chosen}
    (outdir / f"{label}_b{budget}.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(label, spent, ceiling)
