#!/usr/bin/env python3
"""Plot one equally weighted point per nonzero adapter."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    with (args.results / "adapters.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values95 = np.asarray([float(row["median_r95_fraction"]) for row in rows])
    values98 = np.asarray([float(row["median_r98_fraction"]) for row in rows])
    order = np.argsort(values95)
    median95, median98 = float(np.median(values95)), float(np.median(values98))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(1, len(values95) + 1)
    axes[0].scatter(x, values95[order], s=20, alpha=.75, color="#4C78A8", label="95% energy")
    axes[0].scatter(x, values98[order], s=20, alpha=.65, color="#F58518", label="98% energy")
    axes[0].axhline(median95, color="#4C78A8", lw=1.6, ls="--")
    axes[0].axhline(median98, color="#F58518", lw=1.6, ls="--")
    axes[0].set(xlabel="Adapters sorted by median $r_{95}/r$", ylabel="Adapter median $r_{95}/r$",
                ylim=(0, 1), title="One point per adapter (equal weight)")
    axes[0].legend(); axes[0].grid(alpha=.2)

    bins = np.linspace(0, 1, 21)
    axes[1].hist(values95, bins=bins, alpha=.65, color="#4C78A8", label=f"95%, median={median95:.3f}")
    axes[1].hist(values98, bins=bins, alpha=.55, color="#F58518", label=f"98%, median={median98:.3f}")
    axes[1].axvline(median95, color="#4C78A8", lw=1.8)
    axes[1].axvline(median98, color="#F58518", lw=1.8)
    axes[1].set(xlabel="Adapter median retained-rank fraction", ylabel="Adapters", xlim=(0, 1),
                title="Adapter-balanced redundancy distribution")
    axes[1].legend(); axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(args.results / "adapter_balanced_r95.png", dpi=240); plt.close(fig)
    print(f"Rendered {len(rows)} equally weighted adapter points at 95% and 98% energy")


if __name__ == "__main__":
    main()
