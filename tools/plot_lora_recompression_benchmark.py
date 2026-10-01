#!/usr/bin/env python3
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("result_dir", type=Path)
args = p.parse_args()
methods = ["svd", "florist", "spectral", "fraq", "flashtsqr"]
rows = []
for method in methods:
    path = args.result_dir / method / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows.extend(csv.DictReader(path.open()))
with (args.result_dir / "results.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
labels = ["Dense SVD", "FLoRIST", "SpecTraL", "FraQ", "FlashTSQR"]
colors = ["#777777", "#4c78a8", "#f58518", "#54a24b", "#e45756"]
times = {m: [float(r["median_ms"]) for r in rows if r["method"] == m] for m in methods}
errors = {m: [] for m in methods}
by_key = defaultdict(dict)
for row in rows:
    key = (row["repo_order"], row["filename"], row["module"])
    by_key[key][row["method"]] = row
for group in by_key.values():
    reference = np.asarray(json.loads(group["svd"]["fingerprint"]))
    scale = max(np.linalg.norm(reference), 1e-30)
    for method in methods:
        value = np.asarray(json.loads(group[method]["fingerprint"]))
        errors[method].append(float(np.linalg.norm(value-reference)/scale))
repo = defaultdict(lambda: defaultdict(list))
for r in rows:
    repo[int(r["repo_order"])][r["method"]].append(float(r["median_ms"]))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
x = np.arange(len(methods))
axes[0].bar(x, [np.median(times[m]) for m in methods], color=colors)
axes[0].set(yscale="log", ylabel="Median time per layer (ms)", title="Recompression latency")
axes[0].set_xticks(x, labels, rotation=25, ha="right")
speed = np.array([np.median(times["fraq"]) / np.median(times[m]) for m in methods])
axes[1].bar(x, speed, color=colors); axes[1].axhline(1, color="black", lw=.8)
axes[1].set(ylabel="Speedup over FraQ", title="End-to-end speedup")
axes[1].set_xticks(x, labels, rotation=25, ha="right")
for m, label, color in zip(methods, labels, colors):
    axes[2].scatter([np.median(times[m])], [max(errors[m])], label=label, color=color, s=55)
axes[2].set(xscale="log", yscale="symlog", xlabel="Median time per layer (ms)",
            ylabel="Max relative error vs dense SVD", title="Speed / numerical agreement")
axes[2].legend(fontsize=8); axes[2].grid(alpha=.25)
fig.tight_layout(); fig.savefig(args.result_dir / "unified_benchmark.png", dpi=220); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.8))
width = .16; repos = sorted(repo)
for j, (m, label, color) in enumerate(zip(methods, labels, colors)):
    ax.bar(np.arange(len(repos)) + (j-2)*width,
           [np.median(repo[i][m]) for i in repos], width, label=label, color=color)
ax.set(yscale="log", xlabel="Top-10 repository order", ylabel="Median layer time (ms)",
       title="Method latency across Top-10 LoRA repositories")
ax.set_xticks(np.arange(len(repos)), repos); ax.legend(ncol=5, fontsize=8)
ax.grid(axis="y", alpha=.25); fig.tight_layout()
fig.savefig(args.result_dir / "latency_by_repo.png", dpi=220); plt.close(fig)
agreement = {m: {"max_fingerprint_rel_error_vs_svd": max(errors[m]),
                 "median_fingerprint_rel_error_vs_svd": float(np.median(errors[m]))}
             for m in methods}
(args.result_dir / "agreement.json").write_text(json.dumps(agreement, indent=2) + "\n")
print(json.dumps({"plots": ["unified_benchmark.png", "latency_by_repo.png"],
                  "agreement": agreement}, indent=2))
