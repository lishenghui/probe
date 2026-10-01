#!/usr/bin/env python3
"""Budget-utility curve: uniform truncation vs strength-calibrated allocation.

One threshold cannot answer whether strength calibration is worth anything,
because a loose budget leaves every adapter intact and a rule that reallocates
rank has nothing to rescue.  The question is where the two rules separate:

    as compression becomes binding, when does allocating by strength start to
    beat spending the same directions uniformly?

Relative drop is not comparable across these tasks -- 0.955 -> 0.910 on SST-2 and
0.510 -> 0.485 on GSM8K are both -5%, but one is a fifth of the way to the floor
and the other a twentieth -- so the headline metric is retention over headroom,

    R_i = (U_i^comp - U_i^base) / (U_i^orig - U_i^base),

reported as mean, worst, p10 and the count of tasks that fall below a threshold.

The floor is the un-adapted base model on the same prompts, not the chance rate of
the task's label set.  Chance is the wrong reference for a generative evaluation:
Mistral-7B already answers a third of GSM8K and half of CoNLL++ with no adapter
attached, so scoring against chance credits the adapter with ability the base model
supplied and hides the regime that matters -- a compressed adapter that scores
*below* the model it was attached to.  Under the base-model floor that regime is
visible as R < 0, and it is where the two rules differ most.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ("#2a78d6", "#eb6834")            # validated all-pairs, light surface
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})
RULES = ("uniform", "sct")


def retention(rec, rule, bases):
    ref = bases.get(rec["adapter"], rec["chance"])
    head = rec["metric_orig"] - ref
    if head <= 0:
        return float("nan")
    return (rec["rules"][rule]["metric"] - ref) / head


def summarise(run, rule, floor, bases):
    R = np.array([retention(r, rule, bases) for r in run["adapters"]])
    return dict(mean=float(R.mean()), worst=float(R.min()),
                p10=float(np.percentile(R, 10)), median=float(np.median(R)),
                below=int((R < floor).sum()),
                # "broken" is no longer "at chance" but "no better than the
                # un-adapted model", which is the condition a deployment cares about
                broken=int((R <= 0).sum()),
                spent=sum(r["rules"][rule]["k"] for r in run["adapters"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--floor", type=float, default=0.9,
                    help="retention below this counts as a damaged task")
    ap.add_argument("--bases", type=Path, nargs="*", default=[],
                    help="runs carrying metric_base under the same decoding; these "
                         "supply the un-adapted floor. Adapters absent from them fall "
                         "back to the task's chance rate.")
    args = ap.parse_args()

    bases = {}
    for f in args.bases:
        for r in json.loads(f.read_text()):
            if isinstance(r.get("metric_base"), (int, float)):
                bases.setdefault(r["adapter"], r["metric_base"])
    print(f"base-model floor for {len(bases)} adapters: "
          + ", ".join(f"{k}={v:.3f}" for k, v in sorted(bases.items())))
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = sorted((json.loads(f.read_text()) for f in args.results if f.is_file()),
                  key=lambda r: -r["budget"])
    if not runs:
        print("no allocation results found")
        return

    print(f"{'budget':>7} {'tau':>5} {'rule':8s} {'spent':>6} {'mean R':>7} {'worst':>7} "
          f"{'p10':>7} {'median':>7} {f'R<{args.floor}':>6} {'broken':>7}")
    for run in runs:
        for rule in RULES:
            s = summarise(run, rule, args.floor, bases)
            print(f"{run['budget']:7d} {run['match_tau']:5.2f} {rule:8s} {s['spent']:6d} "
                  f"{s['mean']:7.3f} {s['worst']:7.3f} {s['p10']:7.3f} {s['median']:7.3f} "
                  f"{s['below']:6d} {s['broken']:7d}")
        print()

    print("per-adapter retention")
    names = [r["adapter"] for r in runs[0]["adapters"]]
    print(f"{'budget':>7} {'rule':8s} " + " ".join(f"{n[:10]:>10}" for n in names))
    for run in runs:
        for rule in RULES:
            print(f"{run['budget']:7d} {rule:8s} " +
                  " ".join(f"{retention(r, rule, bases):10.3f}" for r in run["adapters"]))

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), constrained_layout=True)
    x = [r["budget"] for r in runs]
    # mean, worst and p10 -- the three tail statistics fixed before the sweep ran.
    # The count of tasks below a threshold is in the table but not plotted: SCT
    # spreads mild damage more widely, so that count does not favour it and a
    # panel chosen for its threshold would be cherry-picking either way.
    for ax, key, lab in ((axes[0], "mean", "mean retention"),
                         (axes[1], "worst", "worst-task retention"),
                         (axes[2], "p10", "$p10$ retention")):
        for c, rule in zip(SERIES, RULES):
            y = [summarise(r, rule, args.floor, bases)[key] for r in runs]
            ax.plot(x, y, "-o", color=c, lw=1.4, ms=4, mec="#fcfcfb", mew=0.6,
                    label="uniform $\\tau$" if rule == "uniform" else "SCT")
        ax.axhline(0.0, color=INK2, lw=0.8, ls=":", zorder=1)   # the un-adapted model
        ax.set_xlabel("retained directions (budget)")
        ax.set_ylabel(lab)
        ax.grid(True, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="lower right")
    out = args.outdir / "rq3_budget_utility.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
