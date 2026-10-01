#!/usr/bin/env python3
"""Export every repo-level value used by the final latency figure to one NPZ."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "artifacts/hf_lora_census/unified_benchmark"
run4 = base / "run_all100_4way_2h_20260812"
svd = base / "run_random10_naive_svd_20260813/svd/repo_results.csv"
methods = np.asarray(["svd", "florist", "spectral", "fraq", "flashtsqr"])
paths = {"svd": svd, **{m: run4 / m / "repo_results.csv" for m in methods[1:]}}

payload = {
    "methods": methods,
    "schema_version": np.asarray(1, dtype=np.int64),
    "latency_unit": np.asarray("milliseconds"),
    "timing_scope": np.asarray("compression only; disk I/O and CPU-to-GPU transfer excluded"),
}
means, stds, medians, q1s, q3s = [], [], [], [], []
layer_means, layer_stds, layer_medians = [], [], []
for method in methods:
    rows = list(csv.DictReader(paths[str(method)].open()))
    prefix = str(method)
    order = np.asarray([int(r["repo_order"]) for r in rows], dtype=np.int64)
    repo_id = np.asarray([r["repo_id"] for r in rows])
    layers = np.asarray([int(r["processed_layers"]) for r in rows], dtype=np.int64)
    latency = np.asarray([float(r["latency_ms"]) for r in rows], dtype=np.float64)
    per_layer = latency / layers
    payload[f"{prefix}_repo_order"] = order
    payload[f"{prefix}_repo_id"] = repo_id
    payload[f"{prefix}_processed_layers"] = layers
    payload[f"{prefix}_latency_ms"] = latency
    payload[f"{prefix}_latency_ms_per_layer"] = per_layer
    means.append(latency.mean()); stds.append(latency.std(ddof=1)); medians.append(np.median(latency))
    q1, q3 = np.percentile(latency, [25, 75]); q1s.append(q1); q3s.append(q3)
    layer_means.append(per_layer.mean()); layer_stds.append(per_layer.std(ddof=1)); layer_medians.append(np.median(per_layer))

payload.update({
    "repo_latency_mean_ms": np.asarray(means), "repo_latency_std_ms": np.asarray(stds),
    "repo_latency_median_ms": np.asarray(medians), "repo_latency_q1_ms": np.asarray(q1s),
    "repo_latency_q3_ms": np.asarray(q3s),
    "per_layer_mean_ms": np.asarray(layer_means), "per_layer_std_ms": np.asarray(layer_stds),
    "per_layer_median_ms": np.asarray(layer_medians),
    "sample_description": np.asarray([
        "fixed-seed random 10 repositories; 6776 layers",
        "complete 100-repository census; 54453 layers",
        "complete 100-repository census; 54453 layers",
        "complete 100-repository census; 54453 layers",
        "complete 100-repository census; 54453 layers",
    ]),
})
out = run4 / "five_methods_latency_final.npz"
np.savez_compressed(out, **payload)
print(out)
