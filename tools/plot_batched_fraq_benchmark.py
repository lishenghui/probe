#!/usr/bin/env python3
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("run_dir", type=Path)
a = p.parse_args()
methods = ("fraq", "flashtsqr")
rows = {m: list(csv.DictReader((a.run_dir / m / "results.csv").open())) for m in methods}

def key(r): return (r["repo_order"], r["filename"], r["module"])
indexed = {m: {key(r): r for r in rs} for m, rs in rows.items()}
errors = []
for k, ref in indexed["fraq"].items():
    x = np.asarray(json.loads(ref["fingerprint"])); y = np.asarray(json.loads(indexed["flashtsqr"][k]["fingerprint"]))
    errors.append(np.linalg.norm(x-y) / max(np.linalg.norm(x), 1e-30))

bucket = {}
for m in methods:
    seen = {}
    for r in rows[m]:
        identity = (r["repo_order"], r["out_dim"], r["in_dim"], r["rank"], r["batch_size"])
        seen[identity] = (int(r["batch_size"]), float(r["batch_ms"]))
    bucket[m] = list(seen.values())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for m, label, color in (("fraq", "FraQ (cuSOLVER QR)", "#4c78a8"),
                        ("flashtsqr", "FlashTSQR", "#e45756")):
    x = [v[0] for v in bucket[m]]; y = [v[1] / v[0] for v in bucket[m]]
    axes[0].scatter(x, y, alpha=.75, label=label, color=color)
axes[0].set(xscale="log", yscale="log", xlabel="Batch size", ylabel="Amortized ms / layer",
            title="True batching across homogeneous LoRA layers")
axes[0].grid(alpha=.25); axes[0].legend()

common = defaultdict(lambda: {m: [] for m in methods})
for m in methods:
    for bs, ms in bucket[m]: common[bs][m].append(ms / bs)
sizes = sorted(bs for bs, v in common.items() if v["fraq"] and v["flashtsqr"])
speed = [np.median(common[bs]["fraq"]) / np.median(common[bs]["flashtsqr"]) for bs in sizes]
axes[1].plot(sizes, speed, marker="o", color="#54a24b")
axes[1].axhline(1, color="black", lw=.8)
axes[1].set(xscale="log", xlabel="Batch size", ylabel="FlashTSQR speedup over FraQ",
            title="Speedup by batch size")
axes[1].grid(alpha=.25); fig.tight_layout()
fig.savefig(a.run_dir / "true_batch_comparison.png", dpi=220); plt.close(fig)

summary = {
    "layers": len(rows["fraq"]),
    "max_batch_size": max(int(r["batch_size"]) for r in rows["fraq"]),
    "median_per_layer_ms": {m: float(np.median([float(r["median_ms"]) for r in rows[m]])) for m in methods},
    "median_speedup": float(np.median([float(r["median_ms"]) for r in rows["fraq"]]) /
                              np.median([float(r["median_ms"]) for r in rows["flashtsqr"]])),
    "fingerprint_rel_error": {"median": float(np.median(errors)), "max": float(np.max(errors))},
}
(a.run_dir / "true_batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
