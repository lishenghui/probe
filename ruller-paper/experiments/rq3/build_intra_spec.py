#!/usr/bin/env python3
"""Intra-Spec: keep each adapter's uniform rank, reallocate it inside the adapter.

The rung that separates two things the Uniform -> E2E-Spec step otherwise
confounds. Uniform gives adapter i a total rank K_i^uni implicitly, by applying
one energy threshold to every module; Intra-Spec spends exactly that same
K_i^uni, but distributes it across the adapter's modules by the spectral rule
instead of by a common per-module threshold. No rank moves between adapters, so
the fleet budget and every per-adapter budget are identical to Uniform's.

Whatever it gains is therefore attributable to intra-adapter reallocation alone,
and whatever E2E-Spec gains on top of it is attributable to the fleet-level
reallocation and the measured objective.

The spectral curves are on a coarse k grid, so each adapter takes the largest
grid point not exceeding its uniform rank; the residual is reported so an
unfairly cheap allocation cannot pass unnoticed.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

R = Path("artifacts/rq3/results")
POOLS = {
    "land":   ("grid_alloc_land12_anchor2.json",
               "rank0_spectral512_land12_*.json", [2289, 1321, 943]),
    "cts":    ("grid_alloc_cts_disjoint_anchor2_n25.json",
               "rank0_spectral_output_cts25_*.json", [19202, 8811, 4924]),
    "lorare": ("grid_alloc_lorare_disjoint_anchor2.json",
               "rank0_spectral_lorare_*.json", [5880, 3191, 2683]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=R / "fra_clean_alloc")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for pool, (gridfile, pattern, budgets) in POOLS.items():
        curves = {}
        for path in glob.glob(str(R / pattern)):
            doc = json.loads(Path(path).read_text())
            if "adapter" in doc and "curve" in doc:
                curves[doc["adapter"]] = {int(r["k"]): r for r in doc["curve"]}
        grid = json.loads((R / gridfile).read_text())["budgets"]
        for budget in budgets:
            paired = grid[str(budget)]["paired"]
            # LoRARetriever grids key adapters by the short name, curves by the
            # full FLAN task name; match on the shared prefix
            lookup = dict(curves)
            for full in list(curves):
                lookup.setdefault(re.sub(r"_\d+templates$", "", full), curves[full])
            missing = [n for n in paired if n not in lookup]
            if missing:
                raise SystemExit(f"{pool} b={budget}: no spectral curve for {missing[:3]}")
            alloc, spent, short = {}, 0, 0
            for name, row in paired.items():
                target = int(row["uniform"]["k"])
                usable = [k for k in lookup[name] if k <= target]
                if not usable:
                    raise SystemExit(f"{pool}/{name}: no grid point at or below {target}")
                k = max(usable)
                short += target - k
                spent += k
                alloc[name] = {"k": k, "uniform_k": target,
                               "module_ranks": lookup[name][k]["module_ranks"],
                               "d_js": lookup[name][k]["d_js"]}
            out = args.outdir / f"{pool}_intra_spec_b{budget}.json"
            out.write_text(json.dumps(
                {"pool": pool, "method": "intra-spec", "n": len(alloc),
                 "budget": budget, "spent": spent,
                 "grid_rounding_loss": short, "allocation": alloc}, indent=2) + "\n")
            print(f"  {out.name:30s} n={len(alloc):2d} spent={spent}/{budget} "
                  f"(grid rounding costs {short})")


if __name__ == "__main__":
    main()
