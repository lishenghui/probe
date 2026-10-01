#!/usr/bin/env python3
"""Figure 2 as it would look with the calibration-decomposition panel added.

Draws (a) and (b) exactly as the published two-panel figure does and appends
(c). Written as a separate file on purpose: plot_frontier_comparison.py
reproduces the published figure byte-for-byte and is left untouched.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_frontier_comparison as P
from plot_frontier_comparison import COLORS, INK, MUTED, GRID
from plot_calibration_decomposition import decompose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("artifacts/rq3/results"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    fidelity, matches, adapters = P.collect(args.results)
    rows = decompose(args.results)
    fleets = list(COLORS)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.55), constrained_layout=True)
    rng = np.random.default_rng(7)

    ax = axes[0]
    for i, fleet in enumerate(fleets):
        vals = np.asarray([x["rho"] for x in fidelity if x["fleet"] == fleet])
        parts = ax.violinplot(vals, [i], widths=.72, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(COLORS[fleet]); b.set_edgecolor("none"); b.set_alpha(.16)
        ax.scatter(i + rng.uniform(-.2, .2, len(vals)), vals, s=17, color=COLORS[fleet],
                   alpha=.7, edgecolors="white", linewidths=.3)
        ax.plot([i - .22, i + .22], [np.mean(vals)] * 2, color=COLORS[fleet], lw=2.3)
        ax.text(i, -.13, fr"mean $\rho={np.mean(vals):.3f}$", ha="center", va="top",
                fontsize=8, color=COLORS[fleet])
    ax.set_xticks(range(3), ["LoRA\nLand", "Lots-of-\nLoRAs", "LoRA-\nRetriever"])
    ax.set_ylim(-.18, 1.03)
    ax.set_ylabel(r"within-budget ranking fidelity $\rho_i(K)$")
    ax.set_title("a   The surrogate ranks fixed-budget candidates", loc="left", weight="bold")

    ax = axes[1]
    for fleet in fleets:
        z = [x for x in adapters if x["fleet"] == fleet]
        ax.scatter([x["pred_ratio"] for x in z], [x["measured_ratio"] for x in z], s=30,
                   color=COLORS[fleet], alpha=.8, edgecolors="white", linewidths=.4, label=fleet)
        n = sum(x["measured_ratio"] < 1 for x in z)
        ax.text(.03, .95 - .075 * fleets.index(fleet), f"{fleet}: {n}/{len(z)} spectral-favoring",
                transform=ax.transAxes, color=COLORS[fleet], fontsize=8, va="top")
    lim = (.22, 14)
    ax.plot(lim, lim, color=MUTED, ls=":", lw=1)
    ax.axhline(1, color=INK, ls="--", lw=1); ax.axvline(1, color=INK, ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"surrogate ratio $\widetilde R^S/\widetilde R^F$")
    ax.set_ylabel(r"measured ratio $R^S/R^F$")
    ax.set_title("b   Calibration shifts across proposal shapes", loc="left", weight="bold")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    for i, fleet in enumerate(fleets):
        z = [r for r in rows if r["fleet"] == fleet]
        base = 2 - i
        stats = {}
        for family, key, offset, filled in (("functional", "functional_bias", +.17, True),
                                            ("spectral", "spectral_bias", -.17, False)):
            v = np.asarray([r[key] for r in z])
            stats[family] = np.median(v)
            parts = ax.violinplot(v, [base + offset], widths=.30, vert=False, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(COLORS[fleet] if filled else "none")
                b.set_edgecolor(COLORS[fleet]); b.set_alpha(.30 if filled else 1.0)
                b.set_linewidth(0 if filled else 1.1)
            ax.plot([np.median(v)] * 2, [base + offset - .10, base + offset + .10],
                    color=COLORS[fleet], lw=2.0, solid_capstyle="butt")
        gap = stats["spectral"] - stats["functional"]
        ax.annotate("", xy=(stats["spectral"], base - .015), xytext=(stats["functional"], base - .015),
                    arrowprops=dict(arrowstyle="->", color=COLORS[fleet], lw=1.3, shrinkA=0, shrinkB=0))
        ax.text(stats["spectral"] + .16, base - .015, fr"$\times${math.exp(gap):.2f}",
                ha="left", va="center", fontsize=8.5, color=COLORS[fleet], weight="bold")
        ax.text(-1.65, base, fleet.replace("Lots-of-", "Lots-of-\n").replace("LoRARetriever", "LoRA-\nRetriever"),
                ha="right", va="center", fontsize=8.6, color=COLORS[fleet])
    ax.axvline(0, color=INK, ls="--", lw=1)
    ax.text(0, -.56, "perfectly calibrated", ha="center", va="bottom", fontsize=7.8, color=INK)
    ax.set_yticks([]); ax.set_xlim(-1.7, 7.2); ax.set_ylim(-.72, 2.72)
    ax.set_xlabel(r"surrogate bias  $\log(\widetilde R/R)$   (right = risk overstated)")
    ax.set_title("c   The bias is family-dependent, not random", loc="left", weight="bold")
    ax.plot([], [], color=MUTED, lw=6, alpha=.30, label="functional proposals")
    ax.plot([], [], color=MUTED, lw=1.2, label="spectral proposals")
    ax.legend(loc="upper right", fontsize=8)
    for s in ("left",):
        ax.spines[s].set_visible(False)

    for ax in axes:
        ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {args.output} and .png")


if __name__ == "__main__":
    main()
