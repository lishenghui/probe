#!/usr/bin/env python3
"""Build the clean Uniform -> E2E-Spec -> FuncDP -> FRA ablation.

All non-Uniform rows use the same measured-JS fleet minimax.  E2E-Spec exposes
only spectral inner proposals, FuncDP only functional-DP proposals, and FRA the
per-(adapter, K) lower-JS envelope of both proposal families.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from two_level_allocation import minimax_allocate


ROOT = Path("artifacts/rq3/results")
POOLS = {
    "land": {
        "budgets": [2289, 1321, 943],
        "functional": "functional_dp0_output_land12_*.json",
        "spectral": "rank0_spectral_output_land12_*.json",
    },
    "cts": {
        "budgets": [19202, 8811, 4924],
        "functional": "functional_dp0_output_cts25_*.json",
        "spectral": "rank0_spectral_output_cts25_*.json",
    },
    "lorare": {
        "budgets": [5880, 3191, 2683],
        "functional": "functional_dp0_output_lorare_*.json",
        "spectral": "rank0_spectral_lorare_*.json",
    },
}


def load(pattern: str) -> dict[str, dict]:
    out = {}
    for filename in glob.glob(str(ROOT / pattern)):
        doc = json.loads(Path(filename).read_text())
        name = doc["adapter"]
        if name in out:
            raise ValueError(f"duplicate adapter {name} for {pattern}")
        out[name] = doc
    return out


def envelope(functional: dict, spectral: dict) -> dict[str, list[dict]]:
    if set(functional) != set(spectral):
        raise ValueError("functional/spectral adapter sets differ")
    curves = {}
    for name in sorted(functional):
        by_source = {
            "functional": {int(r["k"]): r for r in functional[name]["curve"]},
            "spectral": {int(r["k"]): r for r in spectral[name]["curve"]},
        }
        budgets = sorted(set(by_source["functional"]) | set(by_source["spectral"]))
        rows = []
        for k in budgets:
            available = [s for s in by_source if k in by_source[s]]
            source = min(available, key=lambda s: float(by_source[s][k]["d_js"]))
            rows.append(dict(by_source[source][k], source=source))
        curves[name] = rows
    return curves


def tagged_curves(docs: dict[str, dict], source: str) -> dict[str, list[dict]]:
    return {name: [dict(row, source=source) for row in doc["curve"]]
            for name, doc in docs.items()}


def main() -> None:
    outdir = ROOT / "fra_clean_alloc"
    outdir.mkdir(exist_ok=True)
    for pool, cfg in POOLS.items():
        functional, spectral = load(cfg["functional"]), load(cfg["spectral"])
        methods = {
            "e2e_spec": tagged_curves(spectral, "spectral"),
            "funcdp": tagged_curves(functional, "functional"),
            "fra": envelope(functional, spectral),
        }
        print(f"{pool}: {len(functional)} adapters")
        for method, curves in methods.items():
            for budget in cfg["budgets"]:
                chosen, ceiling, spent = minimax_allocate(curves, budget)
                doc = {"pool": pool, "method": method, "currency": "d_js",
                       "budget": budget, "spent": spent,
                       "optimal_ceiling": ceiling, "allocation": chosen}
                path = outdir / f"{pool}_{method}_b{budget}.json"
                path.write_text(json.dumps(doc, indent=2) + "\n")
                counts = {s: sum(r["source"] == s for r in chosen.values())
                          for s in ("functional", "spectral")}
                print(f"  {method:8s} B={budget:5d} spent={spent:5d} "
                      f"ceiling={ceiling:.6g} sources={counts}")


if __name__ == "__main__":
    main()
