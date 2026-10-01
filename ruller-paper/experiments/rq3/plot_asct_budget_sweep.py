#!/usr/bin/env python3
"""Plot the matched-budget LoRA Land sweep and the binding allocation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT.parent / "artifacts/rq3/results/grid_alloc_land12_anchor128.json"
OUT = ROOT / "ruller-short/figures/asct_budget_sweep.pdf"
B_MAX = 12 * 512


def summary(paired: dict, method: str) -> tuple[float, float, float]:
    values = np.asarray([row[method]["u"] for row in paired.values()])
    return float(values.mean()), float(np.percentile(values, 10)), float(values.min())


def main() -> None:
    raw = json.loads(RESULT.read_text())
    budgets = sorted((int(b), row) for b, row in raw["budgets"].items())
    x = np.asarray([b / B_MAX for b, _ in budgets] + [1.0])

    colors = {"mean": "#2471A3", "p10": "#E67E22", "worst": "#B03A2E"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.05, 2.65),
                                 gridspec_kw={"width_ratios": [1.0, 1.25]})
    for method, linestyle, label in [("uniform", "--", "Uniform"),
                                      ("sct", "-", "A-SCT")]:
        vals = [summary(row["paired"], method) for _, row in budgets]
        vals.append((1.0, 1.0, 1.0))
        for j, metric in enumerate(("mean", "p10", "worst")):
            ax.plot(x, [v[j] for v in vals], linestyle, marker="o", ms=3.5,
                    lw=1.45, color=colors[metric],
                    label=f"{label} {metric}")
    ax.axhline(0, color="#777777", lw=.7, zorder=0)
    ax.set_xlabel(r"fleet rank budget $B/B_{\max}$")
    ax.set_ylabel("retention $R$")
    ax.set_xlim(.13, 1.03)
    ax.set_ylim(-.36, 1.09)
    ax.grid(alpha=.18, lw=.5)
    ax.legend(fontsize=6.4, ncol=2, loc="lower right", frameon=False,
              columnspacing=.8, handlelength=2.2)

    binding = raw["budgets"]["1321"]["paired"]
    names = sorted(binding, key=lambda n: binding[n]["sct"]["k"] -
                   binding[n]["uniform"]["k"])
    y = np.arange(len(names))
    uniform = [binding[n]["uniform"]["k"] for n in names]
    anchored = [binding[n]["sct"]["k"] for n in names]
    h = .36
    bx.barh(y - h / 2, uniform, h, color="#AAB7B8", label="Uniform")
    bx.barh(y + h / 2, anchored, h, color="#2471A3", label="A-SCT")
    bx.set_yticks(y, [n.replace("glue_", "") for n in names], fontsize=6.4)
    bx.set_xlabel("retained directions $k_i$")
    bx.grid(axis="x", alpha=.18, lw=.5)
    bx.legend(fontsize=7, frameon=False, loc="lower right")
    bx.set_title(r"binding budget ($B=1{,}321$)", fontsize=8)

    fig.subplots_adjust(left=.08, right=.995, bottom=.19, top=.93, wspace=.38)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
