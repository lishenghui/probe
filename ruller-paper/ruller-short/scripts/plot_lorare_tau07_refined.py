"""Plot LoRARetriever score retention under energy-threshold truncation.

The visual language mirrors ``figure_tau07_refined.png``: a baseline band,
muted population curves, and a small number of highlighted vulnerable cases.
Only adapters with at least ``--min-headroom`` improvement over the base model
are included, matching the filtering used in the paper.
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


LEVELS = ["e99", "e95", "e90", "e80", "e70"]
X_LABELS = [r"$\tau=0.99$", r"$\tau=0.95$", r"$\tau=0.90$", r"$\tau=0.80$", r"$\tau=0.70$"]
HIGHLIGHT_STYLES = [
    {"color": "#9333EA", "marker": "d", "ls": ":"},
    {"color": "#DC2626", "marker": "s", "ls": "--"},
    {"color": "#EA580C", "marker": "X", "ls": "--"},
]


def pretty_name(name: str) -> str:
    names = {
        "anli_r2": "ANLI-R2",
        "wnli": "WNLI",
        "multirc": "MultiRC",
    }
    return names.get(name, name.replace("glue_", "").replace("_", " ").title())


def load_rows(pattern: str, min_headroom: float) -> list[dict]:
    rows: list[dict] = []
    for filename in sorted(glob.glob(pattern)):
        with open(filename) as handle:
            rows.extend(json.load(handle))
    rows = [row for row in rows if row.get("headroom", 0.0) >= min_headroom]
    if not rows:
        raise RuntimeError(f"No eligible records found for pattern: {pattern}")
    for row in rows:
        row["retention"] = np.asarray(
            [row["variants"][level]["retained"] for level in LEVELS], dtype=float
        )
    return rows


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.edgecolor": "#94A3B8",
        "axes.linewidth": 0.8,
    })


def plot(rows: list[dict], output: Path) -> None:
    setup_style()
    x = np.arange(len(LEVELS))
    matrix = np.stack([row["retention"] for row in rows])
    median = np.median(matrix, axis=0)
    worst = sorted(rows, key=lambda row: row["retention"][-1])[:3]
    worst_ids = {id(row) for row in worst}

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.axhspan(0.95, 1.05, color="#F8FAFC", alpha=0.95, zorder=0)
    ax.axhline(1.0, color="#64748B", linewidth=1.0, linestyle=":", alpha=0.85, zorder=1)
    ax.axhline(0.0, color="#94A3B8", linewidth=0.8, zorder=1)

    # The population is intentionally quiet: 41 individually colored curves
    # would obscure the distribution and make the three vulnerable cases hard
    # to identify.
    first_background = True
    for row in rows:
        if id(row) in worst_ids:
            continue
        ax.plot(
            x, row["retention"], color="#94A3B8", linewidth=1.0, alpha=0.22,
            label=f"Other eligible adapters ($n={len(rows)-3}$)" if first_background else None,
            zorder=2,
        )
        first_background = False

    ax.plot(
        x, median, color="#0F766E", linewidth=2.2, linestyle="-",
        marker="o", markersize=5.3, markeredgecolor="white", markeredgewidth=0.8,
        label=f"Population median ({median[-1]:.0%})", zorder=4,
    )

    for row, style in zip(worst, HIGHLIGHT_STYLES):
        name = pretty_name(row["short"])
        ax.plot(
            x, row["retention"], color=style["color"], linestyle=style["ls"],
            linewidth=2.4, marker=style["marker"], markersize=7.0,
            markerfacecolor=style["color"], markeredgecolor="white",
            markeredgewidth=1.0, label=f"{name} ({row['retention'][-1]:.0%})", zorder=5,
        )

    below_half = int(np.sum(matrix[:, -1] < 0.5))
    ax.annotate(
        rf"Median adapter retains {median[-1]:.0%} at $\tau=0.70$",
        xy=(4, median[-1]), xytext=(2.25, 1.16),
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.1),
        fontsize=8.8, color="#0F766E", fontweight="semibold", ha="left",
    )
    ax.text(
        0.1, 1.235,
        rf"{len(rows)-below_half}/{len(rows)} adapters retain $\geq 50\%$ of their task improvement at $\tau=0.70$",
        color="#334155", fontsize=8.8, fontweight="semibold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.set_xlim(-0.12, len(LEVELS) - 1 + 0.12)
    ax.set_ylim(-0.06, max(1.32, float(matrix.max()) + 0.08))
    ax.set_yticks(np.arange(0.0, 1.21, 0.2))
    ax.set_yticklabels(["0.0 (0%)", "0.2", "0.4", "0.6", "0.8", "1.0 (100%)", "1.2"])
    ax.set_ylabel(r"Task-Improvement Retention  $\Delta m_{\mathrm{comp}} \,/\, \Delta m_{\mathrm{orig}}$", fontweight="bold")
    ax.set_xlabel(r"Spectral Energy Threshold $\tau$ $\longrightarrow$ (Increasing Compression Severity)", fontweight="bold")
    ax.set_title(
        r"LoRARetriever Performance Retention Under LoRA Truncation ($\tau \geq 0.70$)",
        loc="left", fontweight="bold", pad=12,
    )
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7, linestyle="--", alpha=0.75, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
        facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=8.8,
        title=r"Adapters (retention at $\tau=0.70$)", title_fontsize=9.2,
        alignment="left",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob", default="artifacts/rq3/results/lorare_task_s*.json",
        help="Glob for LoRARetriever result shards",
    )
    parser.add_argument("--min-headroom", type=float, default=0.05)
    parser.add_argument(
        "--output", type=Path,
        default=Path("ruller-paper/figures/figure_lorare_tau07_refined"),
    )
    args = parser.parse_args()
    rows = load_rows(args.input_glob, args.min_headroom)
    plot(rows, args.output)
    print(f"Plotted {len(rows)} eligible adapters to {args.output}.pdf and .png")


if __name__ == "__main__":
    main()
