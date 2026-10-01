#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "artifacts/hf_lora_census/unified_benchmark"
single = base / "run_top10_5way_20260812"
batched = base / "run_top10_truebatch_20260812"
methods = ["svd", "florist", "spectral", "fraq", "flashtsqr"]
labels = ["Dense SVD", "FLoRIST", "SpecTraL", "FraQ\n(batched)", "FlashTSQR\n(batched)"]
colors = ["#777777", "#4c78a8", "#f58518", "#54a24b", "#e45756"]

values, residuals = [], []
for method in methods[:3]:
    x = json.loads((single / method / "summary.json").read_text())["methods"][method]
    values.append(x["median_layer_ms"]); residuals.append(x["median_reconstruction_residual"])
for method in methods[3:]:
    x = json.loads((batched / method / "summary.json").read_text())["methods"][method]
    values.append(x["median_layer_ms"]); residuals.append(x["median_reconstruction_residual"])

xpos = np.arange(5)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
bars = axes[0].bar(xpos, values, color=colors)
axes[0].set_yscale("log")
axes[0].set_ylabel("Median amortized latency (ms / layer)")
axes[0].set_title("Unified LoRA recompression benchmark (Top-10 HF repos)")
axes[0].set_xticks(xpos, labels)
axes[0].grid(axis="y", alpha=.25)
for bar, value in zip(bars, values):
    axes[0].text(bar.get_x()+bar.get_width()/2, value*1.18, f"{value:.3g}", ha="center", fontsize=9)

speedups = [values[0] / value for value in values]
bars = axes[1].bar(xpos, speedups, color=colors)
axes[1].set_yscale("log")
axes[1].set_ylabel("Speedup over dense SVD")
axes[1].set_title("End-to-end decomposition speedup")
axes[1].set_xticks(xpos, labels)
axes[1].axhline(1, color="black", lw=.8)
axes[1].grid(axis="y", alpha=.25)
for bar, value in zip(bars, speedups):
    axes[1].text(bar.get_x()+bar.get_width()/2, value*1.18, f"{value:.1f}×", ha="center", fontsize=9)

fig.text(.5, .01,
         "SVD/FLoRIST/SpecTraL: single-layer median (29 representative layers).  "
         "FraQ/FlashTSQR: true shape-batched amortized median (578 layers, max batch 64).",
         ha="center", fontsize=8.5, color="#444444")
fig.tight_layout(rect=(0, .055, 1, 1))
out = batched / "unified_five_methods_final.png"
fig.savefig(out, dpi=240); plt.close(fig)

summary = {m: {"median_ms_per_layer": v, "speedup_over_svd": values[0]/v,
               "median_reconstruction_residual": r}
           for m, v, r in zip(methods, values, residuals)}
(batched / "unified_five_methods_final.json").write_text(json.dumps(summary, indent=2)+"\n")
print(out)
print(json.dumps(summary, indent=2))
