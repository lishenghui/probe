#!/usr/bin/env python3
"""The video pool: same retained energy, damage ordered by strength.

Sec. 4.5 previously covered non-text modalities with four adapters at one
threshold, which is too thin to fit anything. This is 33 Wan2.1-T2V-1.3B adapters
from 14 independent uploaders over six thresholds.

The left panel is the claim without a fit in it: at tau = 0.99 every adapter has
discarded essentially the same fraction of its own spectrum, and the resulting
divergence still spans an order of magnitude, ordered by S.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b8b7b2", "#eeedea"
DEEP, FAINT = "#154a8c", "#a9c6e8"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK, "legend.frameon": False,
    "figure.dpi": 200,
})


def tidy(ax):
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    recs = [r for pat in args.results for f in sorted(glob.glob(pat))
            for r in json.loads(Path(f).read_text())]
    print(f"{len(recs)} adapters, {len({r.get('owner') for r in recs})} owners")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)

    ax = axes[0]
    S = np.array([r["S"] for r in recs])
    d = np.array([r["variants"]["e99"]["d_v_mean"] for r in recs])
    L = np.array([r["variants"]["e99"]["L_W"] for r in recs])
    ax.scatter(S, d, s=26, c=DEEP, alpha=0.75, lw=0.5, edgecolor="#fcfcfb", zorder=4)
    lo, hi = S.min() * 0.8, S.max() * 1.25
    k = np.polyfit(np.log(S), np.log(d), 1)
    xs = np.geomspace(lo, hi, 50)
    ax.plot(xs, np.exp(np.polyval(k, np.log(xs))), color=DEEP, lw=1.2, ls="--",
            alpha=0.6, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi)
    ax.set_xlabel("adapter strength $S$")
    ax.set_ylabel("$D_v$ at $\\tau=0.99$")
    ax.set_title("Equal spectral loss, unequal damage", loc="left",
                 fontsize=10, weight="bold")
    ax.text(0.03, 0.94, f"$L_W$ = {L.mean():.3f} $\\pm$ {L.std():.3f} across all "
            f"{len(recs)} adapters", transform=ax.transAxes, fontsize=7.8,
            color=INK2, va="top", style="italic")
    tidy(ax)

    ax = axes[1]
    order = np.argsort([r["S"] for r in recs])
    cmap = plt.get_cmap("viridis")
    for i, j in enumerate(order):
        r = recs[j]
        taus = sorted(r["variants"], key=lambda k: -int(k[1:]))
        ax.plot([int(t[1:]) / 100 for t in taus],
                [r["variants"][t]["d_v_mean"] for t in taus], "-",
                color=cmap(i / max(len(order) - 1, 1)), lw=1.1, alpha=0.85, zorder=3)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("retained energy threshold $\\tau$")
    ax.set_ylabel("$D_v$")
    ax.set_title("33 adapters, 14 uploaders", loc="left", fontsize=10, weight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(math.log10(S.min()), math.log10(S.max())))
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("$\\log_{10} S$", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    tidy(ax)

    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
