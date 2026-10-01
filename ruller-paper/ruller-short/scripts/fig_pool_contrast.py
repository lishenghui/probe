#!/usr/bin/env python3
"""The contrast that makes the threshold's meaninglessness concrete.

Sec. 2.2 shows one threshold doing different things inside a population. This
shows it doing different things *between* populations, which is the version a
practitioner meets: the same tau = 0.70 that leaves 38 LoraRetriever adapters
with a median 0.88 of their utility puts three of ten LoRA Land adapters at or
below the un-adapted model.

The right panel is the answer. Retained energy is matched by construction, so it
cannot explain the split; the model-relative perturbation P = S * L_W is not, and
the two populations do not overlap on it. LoraRetriever's largest P anywhere in
the sweep is smaller than the P at which the weakest LoRA Land adapter fails.

A homogeneous pool is not a wasted pool. It is the control arm: it is what
"nothing happens" looks like, and without it the LoRA Land failures could be
blamed on compression as such rather than on how strong those adapters are.

The left panel plots the 10th percentile, not the median, and the medians are
drawn faintly beside it to show why. On the median the two populations are
indistinguishable, and LoRA Land is in fact slightly *higher* at tau = 0.50,
because seven of its ten adapters barely move. Damage from uncalibrated
truncation is a tail phenomenon; a summary that averages it away reports that
nothing happened right up until a deployment breaks.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAND, RETR = "#eb6834", "#2a78d6"       # validated pair, light surface
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b8b7b2", "#eeedea"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK, "legend.frameon": False,
    "figure.dpi": 200,
})
TAUS = (99, 95, 90, 80, 70, 50)


def tidy(ax):
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--land", nargs="+", required=True)
    ap.add_argument("--retriever", nargs="+", required=True)
    ap.add_argument("--retriever-strength", type=Path, required=True)
    ap.add_argument("--min-headroom", type=float, default=0.05)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    land = {}
    for pat in args.land:
        for f in sorted(glob.glob(pat)):
            for r in json.loads(Path(f).read_text()):
                land[r["adapter"]] = r
    retr = [r for pat in args.retriever for f in sorted(glob.glob(pat))
            for r in json.loads(Path(f).read_text())
            if r["headroom"] >= args.min_headroom]
    S = {e["task"]: e["S"] for e in json.loads(args.retriever_strength.read_text())}

    lu = {t: [] for t in TAUS}          # retained utility
    lp = {t: [] for t in TAUS}          # model-relative perturbation
    for r in land.values():
        fl, hd = r["metric_base"], r["metric_orig"] - r["metric_base"]
        if hd < args.min_headroom:
            continue
        for t in TAUS:
            v = r["variants"][f"e{t}"]
            lu[t].append((v["metric"] - fl) / hd)
            lp[t].append(v["P"])
    ru = {t: [] for t in TAUS}
    rp = {t: [] for t in TAUS}
    for r in retr:
        s = S.get(r["short"])
        for t in TAUS:
            v = r["variants"][f"e{t}"]
            ru[t].append(v["retained"])
            if s:
                rp[t].append(s * v["L_W"])
    print(f"{len(lu[90])} LoRA Land adapters, {len(ru[90])} LoraRetriever adapters")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), constrained_layout=True)
    x = [t / 100 for t in TAUS]

    ax = axes[0]
    for series, colour, lab in ((lu, LAND, "LoRA Land"), (ru, RETR, "LoraRetriever")):
        # the median is the statistic that hides this effect, so it is drawn but
        # subordinated; the bold line is the tail the effect actually lives in
        ax.plot(x, [np.median(series[t]) for t in TAUS], "--", color=colour, lw=1.1,
                alpha=0.55, zorder=3)
        ax.plot(x, [np.percentile(series[t], 10) for t in TAUS], "-o", color=colour,
                lw=2.2, ms=4, mec="#fcfcfb", mew=0.6, label=lab, zorder=4)
    ax.axhline(0.0, color=INK2, lw=1.0, zorder=3)
    ax.text(0.505, 0.04, "un-adapted model", fontsize=7.6, color=INK2,
            ha="left", va="bottom", style="italic")
    ax.annotate("medians (both pools)", (0.60, np.median(lu[70])),
                textcoords="offset points", xytext=(0, -13), ha="center",
                fontsize=7.6, color=INK2, style="italic")
    ax.set_xlabel("retained energy threshold $\\tau$")
    ax.set_ylabel("task utility retained\n(10th percentile, solid)")
    ax.legend(loc="lower left")
    ax.set_ylim(-0.12, 1.14)
    ax.set_title("Same rule, opposite tail", loc="left", fontsize=10, weight="bold")
    tidy(ax); ax.invert_xaxis()

    ax = axes[1]
    for series, colour, lab in ((lp, LAND, "LoRA Land"), (rp, RETR, "LoraRetriever")):
        ax.fill_between(x, [np.percentile(series[t], 10) for t in TAUS],
                        [np.percentile(series[t], 90) for t in TAUS],
                        color=colour, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, [np.median(series[t]) for t in TAUS], "-o", color=colour, lw=2.0,
                ms=4, mec="#fcfcfb", mew=0.6, label=lab, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("retained energy threshold $\\tau$")
    ax.set_ylabel("$P = S\\,L_W$")
    ax.set_title("The quantity the rule ignores", loc="left", fontsize=10, weight="bold")
    tidy(ax); ax.invert_xaxis()

    fig.text(0.5, -0.07, "Both populations are truncated to the same retained energy at "
             "every point. Their medians coincide; their tails and their $P$ do not.",
             ha="center", fontsize=8.2, color=INK2, style="italic")
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
