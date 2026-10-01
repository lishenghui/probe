#!/usr/bin/env python3
"""Compute equally weighted adapter-level statistics from census layer rows."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    with (args.results / "repos.csv").open(newline="") as handle:
        repo_order = {
            row["repo_id"]: int(row["popularity_order"])
            for row in csv.DictReader(handle)
        }
    with (args.results / "layers.csv").open(newline="") as handle:
        layers = list(csv.DictReader(handle))
    spectra_data = np.load(args.results / "spectra.npz")
    spectra = spectra_data["singular_values"]
    spectrum_ranks = spectra_data["nominal_rank"]
    if len(spectra) != len(layers):
        raise ValueError("spectra.npz and layers.csv have different row counts")
    for row, raw, rank in zip(layers, spectra, spectrum_ranks):
        s = raw[:int(rank)].astype(np.float64)
        energy = s * s
        total = energy.sum()
        if total > 0:
            r98 = int(np.searchsorted(np.cumsum(energy) / total, .98) + 1)
        else:
            r98 = 0
        row["r98"] = r98
        row["r98_fraction"] = r98 / int(row["nominal_rank"])

    groups = defaultdict(list)
    for row in layers:
        groups[(row["repo_id"], row["filename"])].append(row)

    adapters, all_zero = [], []
    for (repo_id, filename), rows in groups.items():
        # Energy ranks are undefined for a zero update. Do not turn these into
        # artificial q=0 observations; retain their count for transparency.
        nonzero = [row for row in rows if float(row["stable_rank"]) > 0]
        if not nonzero:
            all_zero.append({
                "repo_id": repo_id, "filename": filename,
                "layers": len(rows), "reason": "all selected-rank layers have zero spectral energy",
            })
            continue
        adapter = {
            "repo_order": repo_order[repo_id], "repo_id": repo_id, "filename": filename,
            "layers_total": len(rows), "layers_nonzero": len(nonzero),
            "layers_zero": len(rows) - len(nonzero),
            "nominal_ranks": ";".join(sorted({row["nominal_rank"] for row in nonzero}, key=int)),
        }
        for level in (90, 95, 98, 99):
            values = np.asarray([float(row[f"r{level}_fraction"]) for row in nonzero])
            adapter[f"median_r{level}_fraction"] = float(np.median(values))
            adapter[f"q25_r{level}_fraction"] = float(np.quantile(values, .25))
            adapter[f"q75_r{level}_fraction"] = float(np.quantile(values, .75))
        adapters.append(adapter)
    adapters.sort(key=lambda row: (row["repo_order"], row["filename"]))

    summary = {
        "statistical_unit": "adapter (repo_id + SafeTensors filename)",
        "aggregation": "median across nonzero layers within adapter, then median across adapters",
        "adapters_included": len(adapters),
        "repositories_represented": len({row["repo_id"] for row in adapters}),
        "all_zero_adapters_excluded": len(all_zero),
        "zero_layers_excluded_within_nonzero_adapters": sum(row["layers_zero"] for row in adapters),
    }
    for level in (90, 95, 98, 99):
        values = np.asarray([row[f"median_r{level}_fraction"] for row in adapters])
        summary[f"adapter_balanced_median_r{level}_fraction"] = float(np.median(values))
        summary[f"adapter_balanced_q25_r{level}_fraction"] = float(np.quantile(values, .25))
        summary[f"adapter_balanced_q75_r{level}_fraction"] = float(np.quantile(values, .75))

    write_csv(args.results / "adapters.csv", adapters)
    if all_zero:
        write_csv(args.results / "all_zero_adapters_excluded.csv", all_zero)
    else:
        (args.results / "all_zero_adapters_excluded.csv").write_text(
            "repo_id,filename,layers,reason\n"
        )
    (args.results / "adapter_balanced_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
