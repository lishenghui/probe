#!/usr/bin/env python3
"""Fixed-budget fidelity and cross-proposal surrogate calibration (Figure 2).

Data logic is unchanged from the reproduction bundle; only the drawing is
restyled to match Figure 1 (scripts/rq1_rank_utilization.py): the canvas is the
final printed width, so the source is never down-scaled, and the type sizes,
palette, grid and legend follow that figure.
"""
from __future__ import annotations
import argparse, csv, glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Figure 1's palette, reused so the two figures read as a pair.
COLORS = {"LoRA Land": "#2864DC", "Lots-of-LoRAs": "#15966A",
          "LoRARetriever": "#E12D2D"}
INK, MUTED = "#171717", "#61728A"

CONFIG = [
 ("LoRA Land", "fixedrho_land_*.json", "functional_dp0_output_land12_*.json",
  "rank0_spectral512_land12_*.json"),
 ("Lots-of-LoRAs", "fixedrho_cts_*.json", "functional_dp0_output_cts25_*.json",
  "rank0_spectral_output_cts25_*.json"),
 ("LoRARetriever", "fixedrho_lorare_*.json", "functional_dp0_output_lorare_*.json",
  "rank0_spectral_lorare_*.json")]


def load(root: Path, pat: str):
    out = {}
    for p in glob.glob(str(root / pat)):
        d = json.loads(Path(p).read_text()); out[d["adapter"]] = d
    return out


def surrogate(doc, ranks):
    tab = {(int(x["module"]), int(x["k"])): float(x["d_js"]) for x in doc["single_layer"]}
    return sum(tab.get((i, int(k)), 0.) for i, k in enumerate(ranks))


def collect(root: Path, tol=.02):
    fidelity, matches, adapters = [], [], []
    for fleet, mp, fp, sp in CONFIG:
        meta, F, S = load(root, mp), load(root, fp), load(root, sp)
        for name in sorted(meta.keys() & F.keys() & S.keys()):
            d = meta[name]
            for budget in sorted({int(x["budget"]) for x in d["fixed_budget_allocations"]}):
                rows = [x for x in d["fixed_budget_allocations"] if int(x["budget"]) == budget]
                rho = float(spearmanr([surrogate(d, x["module_ranks"]) for x in rows],
                                      [float(x["d_js"]) for x in rows]).statistic)
                fidelity.append(dict(fleet=fleet, adapter=name, budget=budget, rho=rho))
            weights = np.asarray(d["module_costs"])
            spec = []
            for x in S[name]["curve"]:
                ranks = tuple(x["module_ranks"])
                spec.append((int(weights @ ranks), ranks, float(x["d_js"]), surrogate(d, ranks)))
            cells = []
            for x in F[name]["curve"]:
                fr = tuple(x["module_ranks"]); fc = int(weights @ fr)
                if not fc: continue
                sc, sr, sm, spred = min(spec, key=lambda z: (abs(z[0] - fc), z[0]))
                fpred, fm = surrogate(d, fr), float(x["d_js"])
                if abs(sc - fc) / fc > tol or sr == fr or min(spred, fpred, sm, fm) <= 0: continue
                cells.append(dict(fleet=fleet, adapter=name, pred_ratio=spred / fpred,
                                  measured_ratio=sm / fm, cost_ratio=sc / fc))
            matches.extend(cells)
            adapters.append(dict(fleet=fleet, adapter=name,
                                 pred_ratio=float(np.median([x["pred_ratio"] for x in cells])),
                                 measured_ratio=float(np.median([x["measured_ratio"] for x in cells])),
                                 cells=len(cells)))
    return fidelity, matches, adapters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("artifacts/rq3/results"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--height", type=float, default=2.05)
    ap.add_argument("--width", type=float, default=3.60)
    ap.add_argument("--compact", action="store_true",
                    help="short labels for a half-column wrapfigure")
    args = ap.parse_args()
    fidelity, matches, adapters = collect(args.results)
    fleets = list(COLORS)

    plt.rcParams.update({
        "font.size": 7.0, "axes.titlesize": 7.5, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 5.8,
        "axes.linewidth": 0.7, "axes.titlepad": 2.0,
    })
    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.height),
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01)
    rng = np.random.default_rng(7)

    ax = axes[0]
    for i, fleet in enumerate(fleets):
        vals = np.asarray([x["rho"] for x in fidelity if x["fleet"] == fleet])
        parts = ax.violinplot(vals, [i], widths=.72, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(COLORS[fleet]); b.set_edgecolor("none"); b.set_alpha(.18)
        ax.scatter(i + rng.uniform(-.2, .2, len(vals)), vals, s=5,
                   color=COLORS[fleet], alpha=.7, edgecolors="white", linewidths=.2)
        ax.plot([i - .24, i + .24], [np.mean(vals)] * 2, color=COLORS[fleet], lw=1.6)
        ax.text(i, 1.12, f"{np.mean(vals):.3f}", ha="center", va="bottom",
                fontsize=5.8, color=COLORS[fleet])
    ax.set_xticks(range(3), ["Land", "Lots", "Retr."] if args.compact
                  else ["LoRA\nLand", "Lots-of-\nLoRAs", "LoRA-\nRetriever"])
    for tick, fleet in zip(ax.get_xticklabels(), fleets):
        tick.set_color(COLORS[fleet])
    ax.set(ylim=(-.12, 1.30),
           ylabel=r"$\rho_i(K)$" if args.compact else r"Fidelity $\rho_i(K)$",
           title="(a) Fidelity" if args.compact else "(a) Ranking fidelity")
    ax.set_yticks((0, .5, 1.0))
    ax.tick_params(axis="y", length=0, pad=1.0)
    ax.tick_params(axis="x", pad=1.2, length=2.0)
    ax.yaxis.labelpad = 0.5
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    # Both axes share one scale, 10^-1 to 10^1.  Two of the 78 adapters fall
    # beyond it; they are pinned just inside the frame as outward triangles
    # rather than dropped, so the count in the legend still matches the cloud.
    LO, HI = .1, 10.
    EDGE = HI / 1.06
    for fleet in fleets:
        z = [x for x in adapters if x["fleet"] == fleet]
        n = sum(x["measured_ratio"] < 1 for x in z)
        inside = [x for x in z if x["pred_ratio"] <= HI and x["measured_ratio"] <= HI]
        ax.scatter([x["pred_ratio"] for x in inside], [x["measured_ratio"] for x in inside],
                   s=8, color=COLORS[fleet], alpha=.85, edgecolors="white",
                   linewidths=.25, label=f"{n}/{len(z)}")
        for x in z:
            if x["pred_ratio"] <= HI and x["measured_ratio"] <= HI:
                continue
            px, py = min(x["pred_ratio"], EDGE), min(x["measured_ratio"], EDGE)
            marker = ">" if x["pred_ratio"] > HI else "^"
            ax.scatter([px], [py], s=11, marker=marker, color=COLORS[fleet],
                       alpha=.85, edgecolors="white", linewidths=.25, clip_on=False)
    xlim = ylim = (LO, HI)
    ax.plot([LO, HI], [LO, HI], color=MUTED, ls=":", lw=.8)
    ax.axhline(1, color=INK, ls="--", lw=.7)
    ax.axvline(1, color=INK, ls="--", lw=.7)
    ax.set(xscale="log", yscale="log", xlim=xlim, ylim=ylim,
           xlabel=r"$\widetilde R^S/\widetilde R^F$" if args.compact
                  else r"Surrogate ratio $\widetilde R^S/\widetilde R^F$",
           ylabel=r"$R^S/R^F$" if args.compact else r"Measured ratio $R^S/R^F$",
           title="(b) Calibration" if args.compact else "(b) Cross-proposal calibration")
    ax.tick_params(axis="y", direction="in", which="both", length=2.2, pad=1.0)
    ax.tick_params(axis="x", pad=1.2, length=2.0)
    ax.yaxis.labelpad = 0.5
    ax.xaxis.labelpad = 0.5
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", frameon=True, framealpha=.9, handlelength=1.1,
              borderpad=0.18, labelspacing=0.1, handletextpad=0.35)

    # (a) is categorical, so a square box just keeps it from looking squashed.
    # (b) is log-log: 'equal' makes one decade the same length on both axes, so
    # the y=x reference really is at 45 degrees.
    axes[0].set_box_aspect(1)
    axes[1].set_box_aspect(1)   # both axes span the same two decades, so 1:1
                                # already puts the y=x reference at 45 degrees

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    with args.output.with_suffix(".csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(matches[0]), lineterminator="\n")
        w.writeheader(); w.writerows(matches)
    print("fixed-budget means",
          {f: np.mean([x['rho'] for x in fidelity if x['fleet'] == f]) for f in fleets})
    print("adapter spectral-favoring",
          {f: sum(x['measured_ratio'] < 1 for x in adapters if x['fleet'] == f) for f in fleets})


if __name__ == "__main__":
    main()
