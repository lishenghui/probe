#!/usr/bin/env python3
"""Score the dense exact allocation against the grid rules on the same budget.

The twelve-level grid and the dense allocator solve the same minimax problem over
different action sets, so the comparison isolates what the action set costs. On the
seventeen-adapter pool the grid version cannot spend precisely: DBpedia's cheapest
upgrade jumps it from 139 to 403 directions, which exhausts the budget and leaves
the two adapters that actually break sitting at their uniform allocation.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/rq3/results")


def stats(u):
    u = np.array(u)
    return dict(mean=float(u.mean()), p10=float(np.percentile(u, 10)),
                worst=float(u.min()), broken=int((u <= 0).sum()))


def main() -> None:
    grid = {r["adapter"]: r for r in
            json.load(open(ROOT / "land17_task_grid.json"))}
    paired = json.load(open(ROOT / "grid_alloc_land17_anchor2.json"))["budgets"]["1973"]["paired"]
    sct = json.load(open(ROOT / "grid_alloc_land17_original.json"))["budgets"]["1973"]["paired"]
    dense = {}
    for f in glob.glob(str(ROOT / "land17_dense_exact_*.json")):
        for r in json.load(open(f)):
            fl = r["metric_base"]
            hd = r.get("headroom", r["metric_orig"] - fl)
            v = list(r["variants"].values())[0]
            dense[r["adapter"]] = dict(k=int(round(v["rank_frac"] * 512)),
                                       u=(v["metric"] - fl) / hd)
    names = sorted(paired)
    missing = [n for n in names if n not in dense]
    if missing:
        print(f"still waiting on: {missing}")
        return
    rows = {"Uniform": [paired[n]["uniform"]["u"] for n in names],
            "SCT (grid)": [sct[n]["sct"]["u"] for n in names],
            "A-SCT (grid)": [paired[n]["sct"]["u"] for n in names],
            "A-SCT (dense exact)": [dense[n]["u"] for n in names]}
    spent = {"Uniform": sum(paired[n]["uniform"]["k"] for n in names),
             "SCT (grid)": sum(sct[n]["sct"]["k"] for n in names),
             "A-SCT (grid)": sum(paired[n]["sct"]["k"] for n in names),
             "A-SCT (dense exact)": sum(dense[n]["k"] for n in names)}
    print(f"LoRA Land, N={len(names)}, budget 1973 (uniform tau=.70)\n")
    print(f"{'rule':22s}{'mean':>8}{'p10':>8}{'worst':>8}{'broken':>8}{'spent':>8}")
    for k, v in rows.items():
        s = stats(v)
        print(f"{k:22s}{s['mean']:8.3f}{s['p10']:8.3f}{s['worst']:8.3f}"
              f"{s['broken']:8d}{spent[k]:8d}")
    print(f"\n{'adapter':22s}{'uni':>7}{'SCT':>8}{'A-SCT':>8}{'dense':>8}   dense k")
    for n in sorted(names, key=lambda x: paired[x]["uniform"]["u"]):
        print(f"{n[:21]:22s}{paired[n]['uniform']['u']:7.3f}{sct[n]['sct']['u']:8.3f}"
              f"{paired[n]['sct']['u']:8.3f}{dense[n]['u']:8.3f}{dense[n]['k']:10d}")


if __name__ == "__main__":
    main()
