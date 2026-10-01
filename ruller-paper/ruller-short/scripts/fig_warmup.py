#!/usr/bin/env python3
"""Warm-up figure for Sec. 2.2: equal compression, unequal outcomes.

Two rules are in common use -- give every adapter the same rank, or give every
adapter the same retained energy -- and both are computed from the adapter alone.
Each is applied here to a controlled pool, and the result is the spread.

Deliberately, adapter strength appears nowhere: the figure is the puzzle, not the
answer, and putting S on an axis would give the answer away before Sec. 2.3 has
introduced it.

Left, the visceral version, on the population where task metrics are recoverable.
Raw accuracy drops are not comparable across seven different tasks, so the bars
are retained utility,

    u = (m_comp - m_base) / (m_full - m_base),

the share of what the adapter added over the un-adapted model that survives
compression. This is a motivating observation only; every later analysis uses a
functional divergence, which does not mix perturbation with task difficulty.

Right, the same two rules on the 32-adapter controlled pool, scored by that
divergence, so the spread is visible in one unit with no task metric involved.
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

BLUE, ORANGE, DEEP = "#2a78d6", "#eb6834", "#184f95"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b8b7b2", "#e8e7e3"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
    "legend.frameon": False, "figure.dpi": 200,
})
PRETTY = {"glue_qqp": "QQP", "glue_sst2": "SST-2", "glue_qnli": "QNLI",
          "hellaswag": "HellaSwag", "viggo": "ViGGO", "gsm8k": "GSM8K",
          "wikisql": "WikiSQL"}


def land(path: Path, labels):
    """Retained utility per adapter under each rule, over the un-adapted floor."""
    out = []
    for r in json.loads(path.read_text()):
        base = r.get("metric_base")
        floor = base if base is not None and np.isfinite(base) else r["chance"]
        head = r["metric_orig"] - floor
        if head <= 0 or any(k not in r["variants"] for k in labels):
            continue
        out.append((PRETTY.get(r["adapter"], r["adapter"]),
                    [(r["variants"][k]["metric"] - floor) / head for k in labels]))
    return out


def pool(pattern: str, labels):
    recs = []
    for f in sorted(glob.glob(pattern)):
        recs += json.loads(Path(f).read_text())
    keep = [r for r in recs if all(k in r["variants"] for k in labels)]
    return [[r["variants"][k]["d_js_mean"] for r in keep] for k in labels], keep


def bars_utility(ax, rows, idx, title, sub):
    names = [n for n, _ in rows]
    vals = np.array([v[idx] for _, v in rows])
    colours = [ORANGE if v < 0.9 else BLUE for v in vals]
    ax.bar(range(len(vals)), vals, color=colours, width=0.66, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=":", zorder=2)
    ax.axhline(0.0, color=INK2, lw=0.9, zorder=2)     # the un-adapted model
    for i, v in enumerate(vals):
        # a bar below zero means the compressed adapter is worse than no adapter
        ax.text(i, v + (0.035 if v >= 0 else -0.035), f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.6,
                color=colours[i], weight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=7.8)
    lo = min(vals.min(), 0.0)
    ax.set_ylim(lo - 0.18 if lo < 0 else -0.02, max(1.18, vals.max() + 0.16))
    if lo < 0:
        ax.text(len(vals) - 0.4, -0.09, "worse than\nno adapter", ha="right", va="top",
                fontsize=7.4, color=ORANGE, style="italic")
    ax.set_title(title, loc="left", color=INK, weight="bold", pad=13)
    ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=INK2,
            style="italic")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def bars_divergence(ax, vals, title, sub):
    order = np.argsort(vals)
    v = np.array(vals)[order]
    ax.bar(range(len(v)), v, color=BLUE, width=0.82, lw=0, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks([])
    ax.set_xlabel(f"{len(v)} adapters, sorted", fontsize=8.2)
    ax.set_title(title, loc="left", color=INK, weight="bold", pad=13)
    ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=INK2,
            style="italic")
    ax.text(0.04, 0.93, f"{v[-1] / v[0]:,.0f}$\\times$ spread", transform=ax.transAxes,
            fontsize=9.6, color=DEEP, va="top", weight="bold")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--land", type=Path, required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--land-labels", nargs=2, default=["k04", "e90"])
    ap.add_argument("--pool-labels", nargs=2, default=["k08", "e90"])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = land(args.land, args.land_labels)
    div, keep = pool(args.pool, args.pool_labels)
    print(f"{len(rows)} LoRA Land adapters, {len(keep)} controlled-pool adapters")

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.2), constrained_layout=True)
    rank_frac = np.mean([r["variants"][args.pool_labels[0]]["rank_frac"] for r in keep])
    bars_utility(axes[0], rows, 0, "(a) same rank",
                 "7 adapters, retained utility, rank halved")
    axes[0].set_ylabel("retained utility $u$")
    bars_utility(axes[1], rows, 1, "(b) same threshold",
                 r"the same 7, $\tau=0.90$")
    bars_divergence(axes[2], div[0], "(c) same rank",
                    f"32 adapters, one base model; retained rank {rank_frac:.0%}")
    axes[2].set_ylabel(r"divergence $D_{\mathrm{JS}}$")
    bars_divergence(axes[3], div[1], "(d) same threshold", r"the same 32, $\tau=0.90$")
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
