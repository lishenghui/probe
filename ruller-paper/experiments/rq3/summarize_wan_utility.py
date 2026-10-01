#!/usr/bin/env python3
"""Summarize paired Wan utility metrics and compute retained utility."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--min-headroom", type=float, default=0.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(json.loads(path.read_text()))
    grouped = defaultdict(dict)
    metadata = {}
    for row in rows:
        key = (row["adapter"], row["prompt_index"], row["seed"])
        if row.get("primary_value") is None:
            continue
        grouped[key][row["variant"]] = float(row["primary_value"])
        metadata[row["adapter"]] = {
            "family": row["family"], "primary_metric": row["primary_metric"],
            "higher_is_better": row["higher_is_better"], "repo": row["repo"],
        }

    result = {}
    for adapter, meta in sorted(metadata.items()):
        pairs = [(key, variants) for key, variants in grouped.items() if key[0] == adapter]
        complete = [(key, v) for key, v in pairs if "base" in v and "full" in v]
        direction = 1.0 if meta["higher_is_better"] else -1.0
        headrooms = [direction * (v["full"] - v["base"]) for _, v in complete]
        variants = sorted({variant for _, values in complete for variant in values
                           if variant not in {"base", "full"}})
        summary = dict(meta)
        summary["n_pairs"] = len(complete)
        summary["mean_base"] = float(np.mean([v["base"] for _, v in complete])) if complete else math.nan
        summary["mean_full"] = float(np.mean([v["full"] for _, v in complete])) if complete else math.nan
        summary["mean_headroom"] = float(np.mean(headrooms)) if headrooms else math.nan
        summary["positive_headroom_fraction"] = float(np.mean(np.array(headrooms) > args.min_headroom)) if headrooms else 0.0
        summary["passes_headroom"] = bool(headrooms and np.mean(headrooms) > args.min_headroom
                                           and np.mean(np.array(headrooms) > 0) >= 0.75)
        summary["variants"] = {}
        for variant in variants:
            retained = []
            for _, values in complete:
                if variant not in values:
                    continue
                denominator = values["full"] - values["base"]
                if abs(denominator) > 1e-12:
                    retained.append((values[variant] - values["base"]) / denominator)
            summary["variants"][variant] = {
                "n": len(retained),
                "mean_retained": float(np.mean(retained)) if retained else math.nan,
                "p10_retained": float(np.percentile(retained, 10)) if retained else math.nan,
                "worst_retained": float(np.min(retained)) if retained else math.nan,
            }
        result[adapter] = summary
        print(f"{adapter}: n={len(complete)} headroom={summary['mean_headroom']:+.6f} "
              f"pass={summary['passes_headroom']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
