#!/usr/bin/env python3
"""Solve fleet minimax allocation from measured two-level adapter curves."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from two_level_allocation import minimax_allocate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", nargs="+", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    docs = []
    for pattern in args.curves:
        for path in glob.glob(pattern):
            doc = json.loads(Path(path).read_text())
            # This tool's own output lands beside the curves it reads, so a second
            # run with the same glob picks up the first run's allocation. It has no
            # `adapter` key, so skip it rather than dying on a KeyError.
            if "adapter" not in doc or "curve" not in doc:
                continue
            docs.append(doc)
    if not docs:
        raise SystemExit(f"no curve files matched {args.curves}")
    curves = {doc["adapter"]: doc["curve"] for doc in docs}
    if len(curves) != len(docs):
        raise SystemExit("duplicate adapter curve")
    chosen, ceiling, spent = minimax_allocate(curves, args.budget)
    result = {"pool": "LoRA Land", "method": "two-level-measured-dense",
              "n": len(chosen), "budget": args.budget, "spent": spent,
              "optimal_ceiling": ceiling, "allocation": chosen}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"two-level dense: n={len(chosen)} spent={spent}/{args.budget} "
          f"ceiling={ceiling:.6e}")
    for name, row in sorted(chosen.items()):
        print(f"  {name:14s} K={row['k']:3d} D_JS={row['d_js']:.6e}")


if __name__ == "__main__":
    main()
