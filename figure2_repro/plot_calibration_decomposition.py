#!/usr/bin/env python3
"""Panel (c): where the cross-proposal calibration error actually lives.

Panels (a) and (b) establish that the additive surrogate ranks candidates well
inside one proposal family yet mis-ranks the two families against each other.
This panel decomposes that error. The quantity plotted in (b) satisfies an
identity, not a hypothesis:

    log[ (Rt_S/Rt_F) / (R_S/R_F) ]  =  log(Rt_S/R_S) - log(Rt_F/R_F)

so the cross-family error is exactly the difference between each family's own
surrogate bias. Plotting the two biases separately shows why (a) and (b) can
both be true: a bias that is roughly a common multiplicative offset inside a
family leaves Spearman untouched, but the two offsets differ across families,
and their gap corrupts every cross-family comparison.

Reads the same artifacts as plot_frontier_comparison.py and reuses its loader,
so the matched-cost pairing is identical by construction rather than by copy.
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_frontier_comparison as P
from plot_frontier_comparison import COLORS, INK, MUTED, GRID


def decompose(root: Path, tol: float = .02):
    """One record per matched-cost comparison, carrying both families' biases."""
    rows = []
    for fleet, mp, fp, sp in P.CONFIG:
        meta, F, S = P.load(root, mp), P.load(root, fp), P.load(root, sp)
        for name in sorted(meta.keys() & F.keys() & S.keys()):
            d = meta[name]
            weights = np.asarray(d["module_costs"])
            spec = []
            for x in S[name]["curve"]:
                ranks = tuple(x["module_ranks"])
                spec.append((int(weights @ ranks), ranks, float(x["d_js"]), P.surrogate(d, ranks)))
            for x in F[name]["curve"]:
                fr = tuple(x["module_ranks"])
                fc = int(weights @ fr)
                if not fc:
                    continue
                sc, sr, sm, spred = min(spec, key=lambda z: (abs(z[0] - fc), z[0]))
                fpred, fm = P.surrogate(d, fr), float(x["d_js"])
                # identical acceptance rule to panel (b)
                if abs(sc - fc) / fc > tol or sr == fr or min(spred, fpred, sm, fm) <= 0:
                    continue
                rows.append(dict(fleet=fleet, adapter=name,
                                 functional_bias=math.log(fpred / fm),
                                 spectral_bias=math.log(spred / sm),
                                 cross_family_error=math.log((spred / fpred) / (sm / fm))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("artifacts/rq3/results"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = decompose(args.results)
    fleets = list(COLORS)

    fig, ax = plt.subplots(figsize=(5.0, 3.55), constrained_layout=True)
    for i, fleet in enumerate(fleets):
        z = [r for r in rows if r["fleet"] == fleet]
        base = 2 - i                       # LoRA Land on top, matching (a)'s order
        stats = {}
        # family is encoded by fill vs outline; hue stays reserved for the fleet
        for family, key, offset, filled in (("functional", "functional_bias", +.17, True),
                                            ("spectral", "spectral_bias", -.17, False)):
            v = np.asarray([r[key] for r in z])
            stats[family] = np.median(v)
            parts = ax.violinplot(v, [base + offset], widths=.30, vert=False, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(COLORS[fleet] if filled else "none")
                b.set_edgecolor(COLORS[fleet])
                b.set_alpha(.30 if filled else 1.0)
                b.set_linewidth(0 if filled else 1.1)
            ax.plot([np.median(v)] * 2, [base + offset - .10, base + offset + .10],
                    color=COLORS[fleet], lw=2.0, solid_capstyle="butt")
        # the gap between the two medians is the quantity panel (b) plots
        gap = stats["spectral"] - stats["functional"]
        ax.annotate("", xy=(stats["spectral"], base - .015), xytext=(stats["functional"], base - .015),
                    arrowprops=dict(arrowstyle="->", color=COLORS[fleet], lw=1.3,
                                    shrinkA=0, shrinkB=0))
        ax.text(stats["spectral"] + .16, base - .015,
                fr"$\times${math.exp(gap):.2f}", ha="left", va="center",
                fontsize=8.5, color=COLORS[fleet], weight="bold")
        ax.text(-1.65, base, fleet.replace("Lots-of-", "Lots-of-\n").replace("LoRARetriever", "LoRA-\nRetriever"),
                ha="right", va="center", fontsize=8.6, color=COLORS[fleet])

    ax.axvline(0, color=INK, ls="--", lw=1)
    ax.text(0, -.56, "perfectly calibrated", ha="center", va="bottom", fontsize=7.8, color=INK)
    ax.set_yticks([])
    ax.set_xlim(-1.7, 7.2)
    ax.set_ylim(-.72, 2.72)
    ax.set_xlabel(r"surrogate bias  $\log(\widetilde R/R)$   (right = risk overstated)")
    ax.set_title("c   The bias is family-dependent, not random", loc="left", weight="bold")
    ax.plot([], [], color=MUTED, lw=6, alpha=.30, label="functional proposals")
    ax.plot([], [], color=MUTED, lw=1.2, label="spectral proposals")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(color=GRID, lw=.7, axis="x")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    with args.output.with_suffix(".csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    print(f"{'fleet':16s} {'n':>5s} {'functional':>22s} {'spectral':>22s} {'gap':>14s}")
    for fleet in fleets:
        z = [r for r in rows if r["fleet"] == fleet]
        f = np.asarray([r["functional_bias"] for r in z])
        s = np.asarray([r["spectral_bias"] for r in z])
        g = np.median(s) - np.median(f)
        print(f"{fleet:16s} {len(z):5d} "
              f"{np.median(f):+7.3f} (x{math.exp(np.median(f)):5.2f}) IQR {np.percentile(f,75)-np.percentile(f,25):4.2f} "
              f"{np.median(s):+7.3f} (x{math.exp(np.median(s)):5.2f}) IQR {np.percentile(s,75)-np.percentile(s,25):4.2f} "
              f"{g:+6.3f} (x{math.exp(g):4.2f})")


if __name__ == "__main__":
    main()
