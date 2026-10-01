#!/usr/bin/env python3
"""LoRA Land utility trajectories under one fleet-wide energy threshold."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER = Path(__file__).resolve().parents[2]
RESULT = PAPER.parent / "artifacts/rq3/results/land12_task_grid.json"
OUT = PAPER / "ruller-short/figures/uniform_threshold_failure.pdf"
LEVELS = ["E100", "E99", "E95", "E90", "E80", "E70"]
KEYS = [None, "e99", "e95", "e90", "e80", "e70"]
HIGHLIGHT = {"dbpedia": "#EB6834", "wikisql": "#4A3AA7", "gsm8k": "#1BAF7A"}
PRETTY = {"glue_qqp": "QQP", "glue_sst2": "SST-2", "glue_qnli": "QNLI",
          "glue_cola": "CoLA", "glue_mrpc": "MRPC", "hellaswag": "HellaSwag",
          "viggo": "ViGGO", "gsm8k": "GSM8K", "wikisql": "WikiSQL",
          "conllpp": "CoNLL++", "e2e_nlg": "E2E", "dbpedia": "DBpedia"}


def main() -> None:
    rows = json.loads(RESULT.read_text())
    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(7.05, 2.7))

    endpoints = []
    for row in rows:
        name = row["adapter"]
        floor = float(row["metric_base"])
        headroom = float(row["metric_orig"]) - floor
        ys = [1.0] + [(float(row["variants"][key]["metric"]) - floor) / headroom
                      for key in KEYS[1:]]
        color = HIGHLIGHT.get(name, "#A9C6E8")
        strong = name in HIGHLIGHT
        ax.plot(x, ys, "-o" if strong else "-", color=color,
                lw=2.25 if strong else 1.05, ms=3.7 if strong else 0,
                alpha=1.0 if strong else .72, zorder=4 if strong else 2,
                solid_capstyle="round")
        if strong:
            endpoints.append((ys[-1], name, color))

    label_y = {"gsm8k": -.28, "dbpedia": -.02, "wikisql": .17}
    for y, name, color in endpoints:
        ax.annotate(PRETTY[name], (x[-1], y), xytext=(11, label_y[name] - y),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.2, color=color, weight="bold", annotation_clip=False)

    ax.axhline(1, color="#6B6B68", lw=.8, ls=":", zorder=1)
    ax.axhline(0, color="#333333", lw=.9, zorder=1)
    ax.fill_between([-.1, len(x)-.9], -.38, 0, color="#FDEDEC", zorder=0)
    ax.text(.05, -.31, "worse than no adapter", fontsize=7, color="#9B2C20",
            style="italic")
    ax.set_xticks(x, LEVELS)
    ax.set_xlabel("more compression  $\longrightarrow$")
    ax.set_ylabel("task utility retained $R$")
    ax.set_ylim(-.38, 1.13); ax.set_xlim(-.08, len(x)-.66)
    ax.grid(color="#EEEDEA", lw=.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.text(.015, .965, "12 adapters, one threshold per operating point",
            transform=ax.transAxes, va="top", fontsize=7.5, color="#52514E",
            style="italic")
    fig.subplots_adjust(left=.09, right=.91, bottom=.24, top=.98)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
