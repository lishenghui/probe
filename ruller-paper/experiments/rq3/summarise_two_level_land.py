#!/usr/bin/env python3
"""Matched-budget utility summary for measured two-level allocation."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load(pattern: str) -> dict:
    out = {}
    for path in glob.glob(pattern):
        for row in json.loads(Path(path).read_text()):
            if "dense" in row.get("variants", {}):
                out[row["adapter"]] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="artifacts/rq3/results/land_dense_exact_*.json")
    ap.add_argument("--two-level", default="artifacts/rq3/results/two_level_eval_land12_*.json")
    ap.add_argument("--output-level",
                    default="artifacts/rq3/results/two_level_output_eval_land12_*.json")
    ap.add_argument("--squared-output",
                    default="artifacts/rq3/results/squared_output_eval_land12_*.json")
    ap.add_argument("--output", type=Path,
                    default=Path("artifacts/rq3/results/two_level_land12_summary.json"))
    args = ap.parse_args()
    methods = {"dense_asct": load(args.baseline),
               "two_level_prompt_js": load(args.two_level),
               "two_level_output_js": load(args.output_level),
               "squared_output_js": load(args.squared_output)}
    names = sorted(set.intersection(*(set(rows) for rows in methods.values())))
    if not names:
        raise SystemExit("no common evaluated adapters")
    result = {"n": len(names), "methods": {}}
    for method, rows in methods.items():
        per_adapter = {}
        for name in names:
            row = rows[name]
            orig, base = float(row["metric_orig"]), float(row["metric_base"])
            comp = float(row["variants"]["dense"]["metric"])
            retention = (comp - base) / (orig - base) if orig > base else float("nan")
            per_adapter[name] = {"orig": orig, "base": base, "compressed": comp,
                                 "retention": retention}
        values = np.asarray([r["retention"] for r in per_adapter.values()])
        result["methods"][method] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.quantile(values, .1)), "worst": float(values.min()),
            "per_adapter": per_adapter,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for method, row in result["methods"].items():
        print(f"{method:12s} mean={row['mean']:.3f} median={row['median']:.3f} "
              f"p10={row['p10']:.3f} worst={row['worst']:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
