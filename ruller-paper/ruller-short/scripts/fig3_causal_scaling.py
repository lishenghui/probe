#!/usr/bin/env python3
"""Figure 3: strength is causal, not a proxy for task difficulty.

Scaling an adapter by lambda multiplies S by lambda and leaves the singular
value *ratios* -- hence the cumulative energy curve and the retained rank --
exactly unchanged.  Task, prompts, spectrum, and rank are therefore held fixed
while strength alone varies, which is the intervention the observational fit
cannot perform.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--lw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = json.loads(args.data.read_text())["scaling"]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["adapter"], []).append(r)
    for v in by.values():
        v.sort(key=lambda x: x["lambda"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.15))
    cmap = plt.get_cmap("viridis")
    order = sorted(by, key=lambda k: by[k][0]["S_base"])
    lo, hi = math.log(by[order[0]][0]["S_base"]), math.log(by[order[-1]][0]["S_base"])

    # (a) one line per adapter: within-adapter response to strength alone
    for k in order:
        v = by[k]
        col = cmap((math.log(v[0]["S_base"]) - lo) / (hi - lo) * .85)
        axA.plot([x["S_effective"] for x in v], [x["d_js_mean"] for x in v],
                 "o-", ms=3.4, lw=1.1, color=col)
        axA.scatter([v[2]["S_effective"]], [v[2]["d_js_mean"]], s=42, facecolor="none",
                    edgecolor=col, lw=1.3, zorder=5)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel(r"effective strength $\lambda S$")
    axA.set_ylabel(r"$D_{\mathrm{JS}}$ (nats)")
    axA.set_title(r"(a) $\lambda$-scaling within each adapter", fontsize=9)
    axA.text(.03, .96, "rings mark $\\lambda=1$\nretained rank identical along each line",
             transform=axA.transAxes, fontsize=6.6, va="top", color="#444")

    # (b) causal vs observational exponent
    def fe(groups):
        cx, cy = [], []
        for v in groups:
            xs = [math.log(x["lambda"]) for x in v]
            ys = [math.log(x["d_js_mean"]) for x in v]
            mx, my = st.mean(xs), st.mean(ys)
            cx += [x - mx for x in xs]; cy += [y - my for y in ys]
        mx, my = st.mean(cx), st.mean(cy)
        return sum((x - mx) * (y - my) for x, y in zip(cx, cy)) / sum((x - mx) ** 2 for x in cx)

    groups = list(by.values())
    a_causal = fe(groups)
    rng = np.random.default_rng(3)
    boot = sorted(fe([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                  for _ in range(4000))
    ci_c = (boot[100], boot[3899])
    a_obs, ci_o = 2.06, (1.81, 2.30)

    for y, (val, ci, label, col) in enumerate((
            (a_obs, ci_o, "observational\n(32 adapters, joint fit)", "#3F79B7"),
            (a_causal, ci_c, r"causal ($\lambda$-intervention," "\n8 adapters, task fixed)", "#B00072"))):
        axB.plot(ci, [y, y], color=col, lw=2.6, solid_capstyle="round")
        axB.plot([val], [y], "o", ms=7, color=col)
        axB.text(val, y + .21, f"{val:.2f}", ha="center", fontsize=8.4, color=col)
        axB.text(-0.06, y, label, ha="right", va="center", fontsize=7.6,
                 transform=axB.get_yaxis_transform())
    axB.axvline(0, color="#999", lw=.8, ls=":")
    axB.set_ylim(-.6, 1.6); axB.set_yticks([])
    axB.set_xlim(-0.15, 3.0)
    axB.set_xlabel(r"strength exponent $a$ in $D\propto S^{a}$")
    axB.set_title("(b) the exponent survives intervention", fontsize=9)

    for ax in (axA, axB):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    axB.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}  a_causal={a_causal:.3f} CI [{ci_c[0]:.3f}, {ci_c[1]:.3f}]")


if __name__ == "__main__":
    main()
