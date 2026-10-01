#!/usr/bin/env python3
"""Warm-up figure for Sec. 2.2: equal compression, unequal outcomes.

A snapshot at one setting shows the outcomes differ. Sweeping the setting shows
the thing that matters: the adapters do not degrade together and do not break at
the same place, so no single rank and no single threshold is a shared operating
point for a fleet.

Columns are the two rules, rows are the two ways of scoring them, and both x-axes
run from mild to severe so "further right is more compressed" holds everywhere.
The bottom row is deliberately a mass rather than a set of trajectories: the
reader should see the spread, not follow any one adapter, so individual lines sit
at low alpha behind an interdecile band and a median.

Adapter strength appears nowhere. This is the puzzle, not the answer.

The pool divergences must come from the fp32 sweeps measured after the prompt
truncation fix (cts_hitau / cts_lowtau / cts_poolrank), and two stale sources are
easy to reach for by mistake. In bf16 the smallest divergences sit at the rounding
floor, which lifts the minimum and collapses the spread to 109x from 496x at
tau=0.90. The pre-fix fp32 sweep is wrong the other way, reporting 404x, because
right truncation at max_length=320 collapses most prompts to a shared preamble.
Both understate the paper's own claim, and neither is a property of the adapters.
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

# Orange, violet and aqua: validated all-pairs, light surface. Two warm hues do
# not work here -- orange against red measures dE 7.1 even unsimulated -- so the
# third highlight is cool rather than warm.
HI = {"DBpedia": "#eb6834", "WikiSQL": "#4a3aa7", "GSM8K": "#1baf7a"}
FAINT, BAND, DEEP = "#a9c6e8", "#c9dcf3", "#154a8c"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b8b7b2", "#eeedea"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 10, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK, "legend.frameon": False,
    "figure.dpi": 200,
})
PRETTY = {"glue_qqp": "QQP", "glue_sst2": "SST-2", "glue_qnli": "QNLI",
          "hellaswag": "HellaSwag", "viggo": "ViGGO", "gsm8k": "GSM8K",
          "wikisql": "WikiSQL", "conllpp": "CoNLL++", "e2e_nlg": "E2E",
          "dbpedia": "DBpedia"}


def read(patterns):
    recs, seen = [], {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for r in json.loads(Path(f).read_text()):
                if r["adapter"] in seen:
                    seen[r["adapter"]]["variants"].update(r["variants"])
                else:
                    seen[r["adapter"]] = dict(r)
                    recs.append(seen[r["adapter"]])
    return recs


def series(rec, prefix):
    """(control, label) ascending, so index 0 is always the most severe setting."""
    out = [(int(k[1:]) / (100 if prefix == "e" else 1), k)
           for k in rec["variants"]
           if k.startswith(prefix) and k != "e100"]   # e100 is the uncompressed control
    return sorted(out)


def tidy(ax):
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.invert_xaxis()              # severity rises to the right in both columns


def panel_utility(ax, rows):
    ends = []
    for name, xs, ys in rows:
        colour = HI.get(name)
        ax.plot(xs, ys, "-", color=colour or FAINT, lw=2.3 if colour else 1.2,
                alpha=1.0 if colour else 0.7, zorder=4 if colour else 3,
                solid_capstyle="round")
        if colour:
            ax.plot(xs, ys, "o", color=colour, ms=3.6, mec="#fcfcfb", mew=0.6, zorder=5)
            ends.append((ys[0], xs[0], name, colour))
    # several highlighted curves finish near u = 0; push their labels apart
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 0.14:
            ends[i] = (ends[i - 1][0] + 0.14,) + ends[i][1:]
    for y, x, name, colour in ends:
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(9, 0),
                    ha="left", va="center", fontsize=9.6, color=colour,
                    weight="bold", annotation_clip=False)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=":", zorder=2)
    ax.axhline(0.0, color=INK2, lw=1.0, zorder=2)
    ax.set_ylim(-0.42, 1.16)
    tidy(ax)


def panel_divergence(ax, curves, ref, prefix):
    grid = curves[0][0]
    band = np.array([ys for _, ys in curves])
    for _, ys in curves:
        ax.plot(grid, ys, "-", color=FAINT, lw=0.8, alpha=0.16, zorder=2)
    ax.fill_between(grid, np.percentile(band, 10, axis=0),
                    np.percentile(band, 90, axis=0), color=BAND, alpha=0.55, lw=0,
                    zorder=3)
    ax.plot(grid, np.median(band, axis=0), color=DEEP, lw=2.4, zorder=4)
    ax.set_yscale("log")
    j = int(np.argmin(np.abs(np.asarray(grid) - ref)))
    row = band[:, j]
    ax.axvline(grid[j], color=MUTED, lw=0.9, ls="--", zorder=2)
    ax.text(0.04, 0.97, f"{row.max() / row.min():,.0f}$\\times$",
            transform=ax.transAxes, fontsize=23, color=DEEP, weight="bold", va="top")
    ax.text(0.04, 0.75, "variation at the same\n"
            + ("rank" if prefix == "k" else "threshold"),
            transform=ax.transAxes, fontsize=9.6, color=DEEP, va="top")
    tidy(ax)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--land", nargs="+", required=True)
    ap.add_argument("--floors", type=Path, nargs="+", required=True,
                    help="runs carrying metric_base; merged, later files win")
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--min-headroom", type=float, default=0.05,
                    help="drop an adapter whose uncompressed score barely clears the "
                         "un-adapted model: the retention denominator is then tiny and "
                         "the ratio is dominated by noise. Sampling GSM8K at T=1 is the "
                         "case this exists for -- a 256-token chain sampled hot loses "
                         "almost all of the adapter's advantage before any compression.")
    ap.add_argument("--ref-rank", type=float, default=8)
    ap.add_argument("--ref-tau", type=float, default=0.90)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    floors = {}
    for f in args.floors:
        floors.update({r["adapter"]: r["metric_base"]
                       for r in json.loads(f.read_text())
                       if r.get("metric_base") is not None
                       and np.isfinite(r["metric_base"])})
    land, pool = read(args.land), read(args.pool)
    print(f"{len(land)} LoRA Land adapters, {len(pool)} pool adapters")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 5.7), constrained_layout=True)
    for col, (prefix, ref, head) in enumerate(
            (("k", args.ref_rank, "Fixed retained rank"),
             ("e", args.ref_tau, "Fixed energy threshold"))):
        rows = []
        for r in land:
            pts = series(r, prefix)
            if len(pts) < 2 or r["adapter"] not in floors:
                continue
            fl = floors[r["adapter"]]
            hd = r["metric_orig"] - fl
            if hd < args.min_headroom:
                print(f"  dropped {r['adapter']}: headroom {hd:.3f} < {args.min_headroom}")
                continue
            rows.append((PRETTY.get(r["adapter"], r["adapter"]),
                         [x for x, _ in pts],
                         [(r["variants"][lab]["metric"] - fl) / hd for _, lab in pts]))
        panel_utility(axes[0][col], rows)
        axes[0][col].set_title(head, loc="left", fontsize=11.5, weight="bold", pad=9)

        curves = []
        for r in pool:
            pts = series(r, prefix)
            if len(pts) < 2:
                continue
            curves.append(([x for x, _ in pts],
                           [r["variants"][lab]["d_js_mean"] for _, lab in pts]))
        panel_divergence(axes[1][col], curves, ref, prefix)
        axes[1][col].set_xlabel("more compression  " + r"$\longrightarrow$")

    axes[0][0].set_ylabel("task utility\nretained")
    axes[1][0].set_ylabel("base-model\ndivergence")
    for ax, nominal in ((axes[0][0], 8), (axes[1][0], 16)):
        ax.text(0.985, 0.04, f"nominal rank $r={nominal}$", transform=ax.transAxes,
                fontsize=8.6, color=INK2, ha="right", va="bottom", style="italic")
    fig.text(0.5, -0.04, f"top: {len(land)} adapters with recoverable task metrics.   "
             "bottom: 32 adapters, one base model, one training pipeline.",
             ha="center", fontsize=9, color=INK2, style="italic")
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
