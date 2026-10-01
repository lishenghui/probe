#!/usr/bin/env python3
"""The chain this paper measures, for Sec. 2.

Lifted out of the old Figure 1, which paired it with the two-adapter comparison
that Figure 1 now carries next to the census. It belongs beside the definitions
rather than in the teaser: the reader needs Eq. (2) and Eq. (3) in front of them
before the missing multiplication means anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

NAVY, MAG, GREY = "#184f95", "#eb6834", "#8a8a8a"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "figure.dpi": 200})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    # ---- right: the measured chain -----------------------------------------
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    boxes = [(0.4, r"$\tau$", "energy\nthreshold", GREY),
             (2.7, r"$L_W$", "adapter-relative\nresidual", GREY),
             (5.4, r"$P=S\,L_W$", "model-relative\nperturbation", NAVY),
             (8.1, r"$D$", "functional\ndivergence", MAG)]
    for x, sym, sub, colour in boxes:
        ax.add_patch(Rectangle((x, 5.4), 1.5, 1.5, facecolor="white",
                                edgecolor=colour, linewidth=1.3))
        ax.text(x + .75, 6.35, sym, ha="center", fontsize=11, color=colour)
        ax.text(x + .75, 5.05, sub, ha="center", va="top", fontsize=7, color="#444")
    for x0, x1 in ((1.9, 2.7), (4.2, 5.4), (6.9, 8.1)):
        ax.add_patch(FancyArrowPatch((x0, 6.15), (x1, 6.15), arrowstyle="-|>",
                                      mutation_scale=11, color="#444", lw=1.0))
    ax.text(4.8, 7.35, r"$\times\,S$", ha="center", fontsize=10, color=NAVY, weight="bold")
    ax.add_patch(FancyArrowPatch((4.8, 7.15), (4.8, 6.6), arrowstyle="-|>",
                                  mutation_scale=10, color=NAVY, lw=1.2))
    ax.text(4.8, 7.9, "the step the literature leaves implicit", ha="center",
             fontsize=7.5, color=NAVY, style="italic")

    ax.text(5.0, 3.35, r"reported: $\tau$, $r_\tau/r$, $L_W$", ha="center", fontsize=8.5, color="#666")
    ax.text(5.0, 2.45, r"responded to: $P$", ha="center", fontsize=8.5, color=NAVY)
    ax.plot([1.15, 4.95], [3.0, 3.0], color="#999", lw=.8, ls=":")
    ax.plot([6.15, 8.85], [2.1, 2.1], color=NAVY, lw=.8, ls=":")
    ax.text(5.0, 1.25, r"$\tau$ is an adapter-relative budget,"
                        r" not a model-relative one",
             ha="center", fontsize=8.6, color=NAVY, weight="bold")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
