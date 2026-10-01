#!/usr/bin/env python3
"""Render publication-style plots from a completed LoRA census."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def curves(spectra, ranks, cumulative, grid):
    output = []
    for raw, rank in zip(spectra, ranks):
        s = raw[:rank].astype(np.float64)
        if cumulative:
            y = np.cumsum(s * s)
            y /= y[-1]
        else:
            y = s / s[0]
        x = np.arange(1, rank + 1) / rank
        output.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
    return np.asarray(output)


def band(ax, grid, values, label, color):
    q25, median, q75 = np.quantile(values, [.25, .5, .75], axis=0)
    ax.fill_between(grid, q25, q75, color=color, alpha=.18)
    ax.plot(grid, median, color=color, linewidth=2.3, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    data = np.load(args.results / "spectra.npz")
    spectra, ranks = data["singular_values"], data["nominal_rank"]
    valid = np.isfinite(spectra[:, 0]) & (spectra[:, 0] > 0)
    removed = int((~valid).sum())
    spectra, ranks = spectra[valid], ranks[valid]
    grid = np.linspace(1 / int(ranks.max()), 1, 256)

    decay = curves(spectra, ranks, False, grid)
    energy = curves(spectra, ranks, True, grid)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for values, ax in zip((decay, energy), axes):
        for curve in values[:1000]:
            ax.plot(grid, curve, color="#4C78A8", alpha=.018, linewidth=.45)
        band(ax, grid, values, "all layers: median ± IQR", "#E45756")
        ax.set_xlim(0, 1); ax.grid(alpha=.2); ax.legend()
        ax.set_xlabel("Retained rank fraction  k/r")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Normalized singular value  $\sigma_k/\sigma_1$")
    axes[0].set_title("LoRA update spectral decay")
    axes[1].set_ylabel("Cumulative squared spectral energy")
    axes[1].set_title("LoRA cumulative energy")
    for level in (.90, .95, .99): axes[1].axhline(level, color="gray", ls="--", lw=.65)
    fig.tight_layout(); fig.savefig(args.results / "spectral_census_overview.png", dpi=240); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {32: "#4C78A8", 64: "#F58518"}
    for rank in sorted(set(ranks.tolist())):
        selected = energy[ranks == rank]
        band(ax, grid, selected, f"rank {rank} (n={len(selected):,})", colors.get(rank, None))
    for level in (.90, .95, .99): ax.axhline(level, color="gray", ls="--", lw=.65)
    ax.set(xlim=(0, 1), xlabel="Retained rank fraction  k/r",
           ylabel="Cumulative squared spectral energy",
           title="Cumulative energy by nominal rank")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(args.results / "cumulative_energy_by_rank.png", dpi=240); plt.close(fig)

    with (args.results / "repos.csv").open(newline="") as handle:
        repos = list(csv.DictReader(handle))
    values = np.asarray([float(row["median_r95_fraction"]) for row in repos])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(values, bins=np.linspace(0, 1, 21), color="#4C78A8", edgecolor="white")
    med = float(np.median(values))
    ax.axvline(med, color="#E45756", lw=2.2, label=f"repository median = {med:.3f}")
    ax.set(xlim=(0, 1), xlabel=r"Repository median $r_{95}/r$", ylabel="Repositories",
           title="Rank fraction required to retain 95% energy")
    ax.grid(axis="y", alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(args.results / "repo_r95_fraction_distribution.png", dpi=240); plt.close(fig)
    print(
        f"Rendered 3 figures from {len(ranks):,} nonzero layers and "
        f"{len(repos)} repositories; excluded {removed:,} all-zero layers"
    )


if __name__ == "__main__":
    main()
