#!/usr/bin/env python3
"""Plot measured safeguarded distortion curves and fleet operating points."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = {
    "sensitivity": "#2471A3",
    "gamma1": "#2471A3",
    "gamma2": "#2E86C1",
    "spectral": "#D35400",
}


def load_curves(pattern: str) -> dict[str, list[dict]]:
    curves = {}
    for filename in glob.glob(pattern):
        doc = json.loads(Path(filename).read_text())
        if "adapter" in doc and "curve" in doc:
            curves[doc["adapter"]] = doc["curve"]
    return curves


def load_allocation(filename: str) -> dict[str, dict]:
    return json.loads(Path(filename).read_text())["allocation"]


def panel(ax, curves, allocation, title, budget):
    grid = sorted(set.intersection(*({int(x["k"]) for x in c} for c in curves.values())))
    full = max(max(int(x["k"]) for x in c) for c in curves.values())
    matrix = []
    for name, curve in sorted(curves.items()):
        rows = {int(x["k"]): x for x in curve}
        x = np.asarray(grid, dtype=float) / full
        y = np.asarray([max(float(rows[k]["d_js"]), 1e-8) for k in grid])
        matrix.append(y)
        ax.plot(x, y, color="#AAB2BD", lw=.65, alpha=.34, zorder=1)
        chosen = allocation[name]
        source = chosen.get("source", "sensitivity")
        ax.scatter(float(chosen["k"]) / full, max(float(chosen["d_js"]), 1e-8),
                   s=19, color=COLORS.get(source, "#2471A3"), edgecolor="white",
                   linewidth=.35, zorder=4)
    ax.plot(np.asarray(grid) / full, np.median(np.asarray(matrix), axis=0),
            color="#17202A", lw=1.8, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(0, 1.01)
    ax.grid(True, which="major", color="#D5D8DC", lw=.45, alpha=.7)
    ax.grid(True, which="minor", axis="y", color="#EAECEE", lw=.3, alpha=.5)
    ax.set_title(title, fontsize=9.5, weight="bold", pad=5)
    ax.text(.98, .94, f"fleet budget = {budget:,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#424949")
    ax.tick_params(labelsize=7.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("ruller-paper/ruller-short/figures/safeguarded_dense_curves"))
    args = ap.parse_args()
    land = load_curves("artifacts/rq3/results/safeguarded_land12_n128_k512/*.json")
    lorare = load_curves("artifacts/rq3/results/safeguarded_lorare/*.json")
    if len(land) != 12 or len(lorare) != 41:
        raise SystemExit(f"incomplete curves: LoRA Land={len(land)}, LoRARetriever={len(lorare)}")

    configs = [
        (land, "artifacts/rq3/results/safeguarded_land12_n128_k512_b2289.json",
         r"LoRA Land: moderate ($\tau_{uni}=.90$)", 2289),
        (land, "artifacts/rq3/results/safeguarded_land12_n128_k512_b1321.json",
         r"LoRA Land: binding ($\tau_{uni}=.70$)", 1321),
        (lorare, "artifacts/rq3/results/safeguarded_lorare_b5880.json",
         r"LoRARetriever: moderate ($\tau_{uni}=.90$)", 5880),
        (lorare, "artifacts/rq3/results/safeguarded_lorare_b3191.json",
         r"LoRARetriever: binding ($\tau_{uni}=.70$)", 3191),
    ]
    plt.rcParams.update({"font.family": "serif", "axes.linewidth": .65})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), sharex=True, sharey="row")
    for ax, (curves, alloc_file, title, budget) in zip(axes.flat, configs):
        panel(ax, curves, load_allocation(alloc_file), title, budget)
    axes[0, 0].set_ylabel(r"E2E JS divergence $D_i(K)$", fontsize=8.5)
    axes[1, 0].set_ylabel(r"E2E JS divergence $D_i(K)$", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("Retained adapter rank fraction", fontsize=8.5)
    legend = [
        Line2D([0], [0], color="#AAB2BD", lw=1, alpha=.65, label="adapter curve"),
        Line2D([0], [0], color="#17202A", lw=1.8, label="fleet median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2471A3",
               markeredgecolor="white", markersize=5, label="sensitivity-selected point"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D35400",
               markeredgecolor="white", markersize=5, label="spectral-selected point"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(.5, -.005))
    fig.tight_layout(rect=(0, .055, 1, 1), w_pad=1.15, h_pad=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
