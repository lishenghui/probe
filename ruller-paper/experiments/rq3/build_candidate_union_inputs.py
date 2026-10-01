#!/usr/bin/env python3
"""Assemble Table-1 method candidates for the cheap labeled union oracle."""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_main_allocation_table import grid_rows, oracle as grid_oracle  # noqa: E402


ROOT = Path("artifacts/rq3/results")
OUT = ROOT / "candidate_union_inputs"


POOLS = {
    "land": {
        "budgets": [2289, 1321, 943],
        "original": "grid_alloc_land12_original.json",
        "anchor": "grid_alloc_land12_anchor2.json",
        "fra_alloc": "funcdp0_alloc/land_fra0_b{budget}.json",
        "fra_eval": "funcdp0_eval_land_*.json",
        "nominal": 512,
        "grid": "artifacts/rq3/results/land12_task_grid.json",
    },
    "cts": {
        "budgets": [19202, 8811, 4924],
        "original": "grid_alloc_cts_original_full_n25.json",
        "anchor": "grid_alloc_cts_disjoint_anchor2_n25.json",
        "fra_alloc": "funcdp0_alloc/cts_fra0_b{budget}.json",
        "fra_eval": "funcdp0_eval_cts_*.json",
        "nominal": 1536,
        "grid": "artifacts/rq3/results/cts_task*.json",
    },
    "lorare": {
        "budgets": [5880, 3191, 2683],
        "original": "grid_alloc_lorare_disjoint_original.json",
        "anchor": "grid_alloc_lorare_disjoint_anchor2.json",
        "fra_alloc": "funcdp0_alloc/lorare_fra0_b{budget}.json",
        "fra_eval": "funcdp0_eval_lorare_*.json",
        "nominal": 512,
        "grid": "artifacts/rq3/results/lorare_task_disjoint_s*.json",
    },
}


def canonical(name: str) -> str:
    return re.sub(r"_10templates$", "", name)


def load_rows(pattern: str) -> list[dict]:
    rows = []
    for filename in glob.glob(str(ROOT / pattern)):
        rows.extend(json.loads(Path(filename).read_text()))
    return rows


def retained(row: dict, variant: dict) -> float:
    if "retained" in variant:
        return float(variant["retained"])
    return ((float(variant["metric"]) - float(row["metric_base"])) /
            (float(row["metric_orig"]) - float(row["metric_base"])))


def fra_scores(pool: str, budget: int) -> dict[str, float]:
    key = f"dense{budget}"
    scores = {}
    for row in load_rows(POOLS[pool]["fra_eval"]):
        if key not in row.get("variants", {}):
            continue
        name = canonical(row.get("short") or row.get("adapter"))
        scores[name] = retained(row, row["variants"][key])
    return scores


def add_2l(pool: str, budget: int, adapters: dict[str, list[dict]]) -> None:
    patterns = {
        ("land", 2289): "land12_2L_b2289_*.json",
        ("land", 1321): "two_level_output_eval_land12_*.json",
        ("cts", 19202): "cts25_2L_b19202_*.json",
        ("cts", 8811): "cts25_2L_b8811_*.json",
        ("lorare", 5880): "lorare_2L_s*.json",
        ("lorare", 3191): "lorare_2L_s*.json",
    }
    pattern = patterns.get((pool, budget))
    if not pattern:
        return
    key = "dense" if pool != "lorare" else f"dense{budget}"
    for row in load_rows(pattern):
        if key not in row.get("variants", {}):
            continue
        name = canonical(row.get("short") or row.get("adapter"))
        if name not in adapters:
            continue
        variant = row["variants"][key]
        adapters[name].append({"k": round(float(variant["rank_frac"]) *
                                           POOLS[pool]["nominal"]),
                               "u": retained(row, variant), "source": "2L A-SCT"})


def build(pool: str, budget: int) -> dict:
    spec = POOLS[pool]
    original = json.loads((ROOT / spec["original"]).read_text())["budgets"][str(budget)]
    anchor = json.loads((ROOT / spec["anchor"]).read_text())["budgets"][str(budget)]
    names = sorted(canonical(x) for x in anchor["paired"])
    adapters = {name: [] for name in names}
    for raw_name, pair in anchor["paired"].items():
        name = canonical(raw_name)
        for source, row in (("Uniform", pair["uniform"]), ("A-SCT", pair["sct"])):
            adapters[name].append({"k": int(row["k"]), "u": float(row["u"]),
                                   "source": source})
    for raw_name, pair in original["paired"].items():
        name = canonical(raw_name)
        adapters[name].append({"k": int(pair["sct"]["k"]),
                               "u": float(pair["sct"]["u"]), "source": "SCT"})

    # Preserve the old labeled threshold-grid oracle as another candidate
    # source.  The union oracle must dominate both it and every deployable rule.
    measured_grid = grid_rows(spec["grid"], spec["nominal"])
    old_oracle = grid_oracle({name: measured_grid[name] for name in names}, budget)
    for name, row in old_oracle.items():
        adapters[name].append({"k": int(row["k"]), "u": float(row["u"]),
                               "source": "Threshold-grid Oracle"})

    allocation = json.loads((ROOT / spec["fra_alloc"].format(budget=budget)).read_text())
    scores = fra_scores(pool, budget)
    for raw_name, row in allocation["allocation"].items():
        name = canonical(raw_name)
        adapters[name].append({"k": int(row["k"]), "u": scores[name],
                               "source": "FRA"})
    add_2l(pool, budget, adapters)
    missing = [name for name, candidates in adapters.items()
               if not any(c["source"] == "FRA" for c in candidates)]
    if missing:
        raise ValueError(f"{pool} B={budget}: missing FRA scores for {missing}")
    return {"pool": pool, "budget": budget, "adapters": adapters,
            "candidate_sources": sorted({c["source"] for cs in adapters.values() for c in cs})}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for pool, spec in POOLS.items():
        for budget in spec["budgets"]:
            doc = build(pool, budget)
            path = OUT / f"{pool}_b{budget}.json"
            path.write_text(json.dumps(doc, indent=2) + "\n")
            counts = sorted({len(x) for x in doc["adapters"].values()})
            print(path, len(doc["adapters"]), doc["candidate_sources"], counts)


if __name__ == "__main__":
    main()
