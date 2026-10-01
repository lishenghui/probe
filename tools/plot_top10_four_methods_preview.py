#!/usr/bin/env python3
import csv, json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / "artifacts/hf_lora_census/unified_benchmark/run_all100_4way_2h_20260812"
methods = ("florist", "spectral", "fraq", "flashtsqr")
labels = ("FLoRIST", "SpecTraL", "FraQ", "FlashTSQR")
colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
data = {m: list(csv.DictReader((run/m/"repo_results.csv").open())) for m in methods}

stats = {}
for m in methods:
    total = np.array([float(r["latency_ms"]) for r in data[m]])
    per_layer = np.array([float(r["latency_ms"])/int(r["processed_layers"]) for r in data[m]])
    q1, med, q3 = np.percentile(total, [25, 50, 75])
    stats[m] = {"mean_ms": float(total.mean()), "std_ms": float(total.std(ddof=1)),
                "median_ms": float(med), "q1_ms": float(q1), "q3_ms": float(q3),
                "min_ms": float(total.min()), "max_ms": float(total.max()),
                "mean_ms_per_layer": float(per_layer.mean()),
                "std_ms_per_layer": float(per_layer.std(ddof=1)),
                "median_ms_per_layer": float(np.median(per_layer))}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
x = np.arange(len(methods))
means = np.array([stats[m]["mean_ms"] for m in methods])
stds = np.array([stats[m]["std_ms"] for m in methods])
# Asymmetric lower errors remain positive on a logarithmic axis.
lower = np.minimum(stds, means * .92)
axes[0].bar(x, means, color=colors, alpha=.88)
axes[0].errorbar(x, means, yerr=np.vstack([lower, stds]), fmt="none", ecolor="black", capsize=5)
repo_count = len(data[methods[0]])
axes[0].set(yscale="log", ylabel="Compression latency / repo (ms)",
            title=f"Mean ± std ({repo_count} repositories)")
axes[0].set_xticks(x, labels, rotation=18, ha="right"); axes[0].grid(axis="y", alpha=.25)

values = [[float(r["latency_ms"]) for r in data[m]] for m in methods]
bp = axes[1].boxplot(values, positions=x, widths=.5, showfliers=False, patch_artist=True)
for box, color in zip(bp["boxes"], colors): box.set_facecolor(color); box.set_alpha(.3)
for j, (vals, color) in enumerate(zip(values, colors)):
    axes[1].scatter(j + np.linspace(-.11, .11, len(vals)), vals, s=24, color=color, alpha=.8)
axes[1].set(yscale="log", ylabel="Compression latency / repo (ms)", title="Distribution across repositories")
axes[1].set_xticks(x, labels, rotation=18, ha="right"); axes[1].grid(axis="y", alpha=.25)

pl_means = np.array([stats[m]["mean_ms_per_layer"] for m in methods])
pl_stds = np.array([stats[m]["std_ms_per_layer"] for m in methods])
pl_lower = np.minimum(pl_stds, pl_means*.92)
axes[2].bar(x, pl_means, color=colors, alpha=.88)
axes[2].errorbar(x, pl_means, yerr=np.vstack([pl_lower, pl_stds]), fmt="none", ecolor="black", capsize=5)
axes[2].set(yscale="log", ylabel="Normalized latency (ms / layer)", title="Repo-size-normalized mean ± std")
axes[2].set_xticks(x, labels, rotation=18, ha="right"); axes[2].grid(axis="y", alpha=.25)

fig.suptitle(f"Top-{repo_count} HF LoRA projects: completed recompression methods", fontsize=14)
fig.tight_layout(); out=run/"four_methods_latency_preview.png"; fig.savefig(out,dpi=240); plt.close(fig)
(run/"four_methods_latency_preview.json").write_text(json.dumps(stats,indent=2)+"\n")
print(out); print(json.dumps(stats,indent=2))
