#!/usr/bin/env python3
"""Figure 1: the premise everyone accepts, and the step it hides.

Left, the setup: across 96,370 LoRA matrices from 200 public repositories, the
median layer reaches 90% of its spectral energy on a third of its nominal rank,
and that is not a property of a few outlier projects -- 183 of the 200 have a
median layer that can drop a quarter of its rank, 142 more than half. Spectral
redundancy is ordinary.

Right, the twist the paper is about: the two extremes of the controlled pool are
truncated to nearly the same adapter-relative residual by the same threshold, and
the base model still receives perturbations 11.6x apart and diverges 382x.  What
tau equalizes is the loss relative to the adapter; what the network experiences
is the loss relative to the weight it perturbs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated all-pairs, light surface
FILL, DEEP = "#9ec5f4", "#184f95"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b8b7b2", "#e8e7e3"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 10.5, "axes.labelsize": 10.5,
    "axes.titlesize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
    "legend.frameon": False, "figure.dpi": 200,
})


def census(root: Path, tau_col: str = "r90"):
    """Removable rank fraction per layer, and each repository's median."""
    per, layers = defaultdict(list), []
    for rank in (32, 64):
        path = root / f"top100_rank{rank}_dominated" / "layers.csv"
        for row in csv.DictReader(path.open(newline="")):
            frac = 1.0 - float(row[tau_col]) / float(row["nominal_rank"])
            per[(rank, row["repo_id"])].append(frac)
            layers.append(frac)
    return np.array(layers), np.array([np.median(v) for v in per.values()])


def panel_premise(ax, layers, repos):
    """Histogram over layers, with one dot per repository in a reserved strip.

    The dots are the point: a median taken over 96,370 layers could be carried by
    a handful of large repositories, and this shows it is not.
    """
    counts, _, _ = ax.hist(layers, bins=np.linspace(0, 1, 41), color=FILL,
                           edgecolor="#fcfcfb", linewidth=0.4, zorder=2)
    top = counts.max() * 1.28
    strip = -top * 0.20                      # reserved band, below every bar
    med = np.median(layers)
    ax.axvline(med, color=DEEP, lw=1.6, ymin=0.16, zorder=4)
    ax.annotate(f"median layer keeps\n{1 - med:.0%} of its rank", (med, top * 0.97),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="top",
                fontsize=10.4, color=DEEP, weight="bold")
    rng = np.random.default_rng(0)
    ax.scatter(repos, strip + rng.normal(0, top * 0.020, repos.size), s=7,
               color=ORANGE, alpha=0.55, lw=0, zorder=3, clip_on=False)
    ax.text(0.015, strip - top * 0.115, "one dot per repository, at its median layer",
            fontsize=9.2, color=ORANGE, va="center", ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(strip - top * 0.17, top)
    ax.set_xlabel(r"rank removable at $\tau=0.90$,   $1-r_\tau/r$")
    ax.set_ylabel("LoRA matrices")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{x:.0%}" for x in np.linspace(0, 1, 6)])
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_position(("data", strip - top * 0.17))
    ax.set_title("Spectral redundancy is ordinary\n"
                 r"$\it{96{,}370\ LoRA\ matrices,\ 200\ public\ repositories}$",
                 loc="left", color=INK, weight="bold", fontsize=12, pad=8)


def panel_twist(ax, weak, strong):
    """Two adapters, one threshold: same adapter-relative loss, different model.

    The per-adapter numbers stack vertically under each bar rather than beside it;
    side labels collide once the type is large enough to read at print size.
    """
    exag = 3.0
    ax.set_xlim(-0.05, 2.15)
    ax.set_ylim(-1.72, 2.15)
    ax.axis("off")
    ax.text(1.05, 2.12, r"one threshold $\tau=0.90$, one base model",
            ha="center", va="top", fontsize=10.0, color=INK2, style="italic")
    for x0, (name, S, L, D, colour) in zip((0.16, 1.32), (weak, strong)):
        delta = S * exag
        ax.add_patch(plt.Rectangle((x0, 0), 0.56, 1.0, facecolor="#ececea",
                                   edgecolor="none", zorder=2))
        ax.text(x0 + 0.28, 0.5, r"$\mathbf{W}$", ha="center", va="center",
                fontsize=13, color="#555", zorder=3)
        ax.add_patch(plt.Rectangle((x0, 1.0), 0.56, delta * (1 - L), facecolor=colour,
                                   edgecolor="none", zorder=3))
        ax.add_patch(plt.Rectangle((x0, 1.0 + delta * (1 - L)), 0.56, delta * L,
                                   facecolor="#fcfcfb", edgecolor=colour, lw=0.9,
                                   hatch="////", zorder=3))
        ax.text(x0 + 0.28, 1.0 + delta + 0.12, name, ha="center", va="bottom",
                fontsize=10.4, weight="bold", color=colour)
        for dy, txt, col, wt in ((-0.20, f"$L_W={L:.3f}$", INK2, "normal"),
                                 (-0.56, f"{100 * S * L:.2f}% of $\mathbf{{W}}$",
                                  colour, "bold"),
                                 (-0.92, f"$D_{{\mathrm{{JS}}}}={D:.6f}$", colour, "bold")):
            ax.text(x0 + 0.28, dy, txt, ha="center", va="top", fontsize=10.0,
                    color=col, weight=wt)
    ax.plot([0.05, 2.05], [-1.30, -1.30], color=GRID, lw=1.0, zorder=1)
    ax.text(1.05, -1.48, r"$\mathbf{11.6\times}$ the perturbation,"
                        r"   $\mathbf{382\times}$ the divergence",
            ha="center", va="center", fontsize=11.2, color=DEEP)
    ax.text(-0.05, 2.15, "…but it is not the same compression", ha="left", va="bottom",
            fontsize=12, color=INK, weight="bold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True,
                    help="directory holding top100_rank{32,64}_dominated/")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    layers, repos = census(args.census)
    print(f"{layers.size:,} layers, {repos.size} repositories; "
          f"median removable {np.median(layers):.1%}; "
          f"repos above 25%/50%: {(repos > .25).sum()}/{(repos > .5).sum()}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.0, 2.85),
                                   gridspec_kw={"width_ratios": [1.12, 1]},
                                   constrained_layout=True)
    panel_premise(axL, layers, repos)
    panel_twist(axR,
                ("weakest adapter", 0.0139, 0.299, 0.000038, BLUE),
                ("strongest adapter", 0.1644, 0.292, 0.014388, ORANGE))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
