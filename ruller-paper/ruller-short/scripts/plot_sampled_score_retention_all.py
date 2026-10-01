"""Plot sampled compressed/original task-score ratio for all ten tasks up to tau=0.70."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "rq3" / "results"
LEVELS = ["e99", "e95", "e90", "e80", "e70"]
X_LABELS = [r"$\tau=0.99$", r"$\tau=0.95$", r"$\tau=0.90$", r"$\tau=0.80$", r"$\tau=0.70$"]

TASKS = {
    "CoNLL-PP": "predibase_samp_conllpp.json",
    "DBpedia": "predibase_samp_dbpedia.json",
    "E2E-NLG": "predibase_samp_e2e_nlg.json",
    "QNLI": "predibase_samp_glue_qnli.json",
    "QQP": "predibase_samp_glue_qqp.json",
    "SST-2": "predibase_samp_glue_sst2.json",
    "HellaSwag": "predibase_samp_hellaswag.json",
    "ViGGO": "predibase_samp_viggo.json",
    "WikiSQL": "predibase_samp_wikisql.json",
}

TASK_CONFIGS = {
    # 3 Degrading / Vulnerable tasks (Highlighted)
    "DBpedia":   {"color": "#9333EA", "marker": "d", "ls": ":",  "lw": 2.4, "ms": 7.0, "alpha": 1.0, "degrading": True},
    "WikiSQL":   {"color": "#DC2626", "marker": "s", "ls": "--", "lw": 2.4, "ms": 6.8, "alpha": 1.0, "degrading": True},
    "GSM8K":     {"color": "#EA580C", "marker": "X", "ls": "--", "lw": 2.4, "ms": 7.0, "alpha": 1.0, "degrading": True},

    # 7 Resilient tasks (Muted, no markers, lighter colors)
    "HellaSwag": {"color": "#38BDF8", "marker": None, "ls": "-", "lw": 1.5, "ms": 0, "alpha": 0.70, "degrading": False},
    "E2E-NLG":   {"color": "#64748B", "marker": None, "ls": "-", "lw": 1.5, "ms": 0, "alpha": 0.70, "degrading": False},
    "SST-2":     {"color": "#14B8A6", "marker": None, "ls": "-", "lw": 1.5, "ms": 0, "alpha": 0.65, "degrading": False},
    "QQP":       {"color": "#60A5FA", "marker": None, "ls": "-", "lw": 1.5, "ms": 0, "alpha": 0.65, "degrading": False},
    "ViGGO":     {"color": "#4ADE80", "marker": None, "ls": "-", "lw": 1.5, "ms": 0, "alpha": 0.65, "degrading": False},
    "CoNLL-PP":  {"color": "#A3E635", "marker": None, "ls": "-", "lw": 1.5, "ms": 0, "alpha": 0.65, "degrading": False},
    "QNLI":      {"color": "#94A3B8", "marker": None, "ls": "-", "lw": 1.5, "ms": 0, "alpha": 0.70, "degrading": False},
}


def load_record(filename: str) -> dict:
    return json.loads((RESULTS / filename).read_text())[0]


def ordinary_curve(filename: str) -> list[float]:
    record = load_record(filename)
    original = record["metric_orig"]
    return [record["variants"][level]["metric"] / original for level in LEVELS]


def gsm8k_curve() -> list[float]:
    locations = {
        "e99": (1, "e99"),
        "e95": (2, "e95"),
        "e90": (3, "e90"),
        "e80": (0, "e80"),
        "e70": (1, "e70"),
    }
    values = []
    for level in LEVELS:
        shard, key = locations[level]
        record = load_record(f"predibase_samp_gsm8k_{shard}.json")
        values.append(record["variants"][key]["metric"] / record["metric_orig"])
    return values


def main() -> None:
    curves = {task: ordinary_curve(filename) for task, filename in TASKS.items()}
    curves["GSM8K"] = gsm8k_curve()

    items = []
    for t, vals in curves.items():
        cfg = TASK_CONFIGS.get(t, {"color": "#475569", "marker": None, "ls": "-", "lw": 1.5, "ms": 0, "alpha": 0.6, "degrading": False})
        items.append({
            "task": t,
            "values": vals,
            "final": vals[-1],
            **cfg
        })

    degrading_tasks = sorted([t for t in items if t["degrading"]], key=lambda x: x["final"])
    resilient_tasks = sorted([t for t in items if not t["degrading"]], key=lambda x: x["final"], reverse=True)

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

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(len(LEVELS))

    # Reference band & baseline
    ax.axhspan(0.95, 1.05, color="#F8FAFC", alpha=0.95, zorder=0)
    ax.axhline(1.0, color="#64748B", linewidth=1.0, linestyle=":", alpha=0.85, zorder=1)
    ax.axhline(0.0, color="#94A3B8", linewidth=0.8, linestyle="-", zorder=1)

    # 1. Plot Resilient tasks in background (no marker, lighter colors)
    for item in resilient_tasks:
        ax.plot(
            x,
            item["values"],
            color=item["color"],
            linestyle=item["ls"],
            linewidth=item["lw"],
            alpha=item["alpha"],
            marker=None,
            label=f"{item['task']} ({item['final']:.0%})",
            zorder=2,
        )

    # 2. Plot Degrading tasks in foreground (bold markers & lines)
    for item in degrading_tasks:
        ax.plot(
            x,
            item["values"],
            color=item["color"],
            linestyle=item["ls"],
            linewidth=item["lw"],
            alpha=item["alpha"],
            marker=item["marker"],
            markersize=item["ms"],
            markerfacecolor=item["color"],
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=f"{item['task']} ({item['final']:.0%})",
            zorder=4,
        )

    # Clean Callout Annotations
    ax.annotate(
        "DBpedia\n(cliff at $\\tau \\leq 0.95$)",
        xy=(1, 0.067), xytext=(1.25, 0.23),
        arrowprops=dict(arrowstyle="->", color=TASK_CONFIGS["DBpedia"]["color"], lw=1.2, shrinkA=2, shrinkB=3),
        fontsize=8.8, color=TASK_CONFIGS["DBpedia"]["color"], fontweight="bold", va="bottom"
    )

    ax.annotate(
        "WikiSQL\n(collapses at $\\tau \\leq 0.80$)",
        xy=(3, 0.810), xytext=(2.05, 0.62),
        arrowprops=dict(arrowstyle="->", color=TASK_CONFIGS["WikiSQL"]["color"], lw=1.2, shrinkA=2, shrinkB=3),
        fontsize=8.8, color=TASK_CONFIGS["WikiSQL"]["color"], fontweight="bold", va="center"
    )

    ax.annotate(
        "GSM8K\n(drops to $46\\%$)",
        xy=(4, 0.463), xytext=(3.30, 0.65),
        arrowprops=dict(arrowstyle="->", color=TASK_CONFIGS["GSM8K"]["color"], lw=1.2, shrinkA=2, shrinkB=3),
        fontsize=8.8, color=TASK_CONFIGS["GSM8K"]["color"], fontweight="bold", va="bottom", ha="left"
    )

    ax.annotate(
        "7 resilient tasks maintain $\\geq 95\\%$ score up to $\\tau=0.70$",
        xy=(3.6, 0.98), xytext=(1.3, 1.10),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.0, shrinkA=2, shrinkB=3),
        fontsize=8.8, color="#334155", fontweight="semibold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.set_xlim(-0.12, len(LEVELS) - 1 + 0.12)
    ax.set_ylim(-0.06, 1.18)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0 (0%)", "0.2", "0.4", "0.6", "0.8", "1.0 (100%)"])

    ax.set_ylabel(r"Task-Score Retention  $m_{\mathrm{comp}} \,/\, m_{\mathrm{orig}}$", fontweight="bold")
    ax.set_xlabel(r"Spectral Energy Threshold $\tau$ $\longrightarrow$ (Increasing Compression Severity)", fontweight="bold")
    ax.set_title(r"Downstream Task Performance Retention Under LoRA Truncation ($\tau \geq 0.70$)", loc="left", fontweight="bold", pad=12)

    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7, linestyle="--", alpha=0.75, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        fontsize=8.8,
        title="Tasks (retention at $\\tau=0.70$)",
        title_fontsize=9.2,
        alignment="left",
    )

    fig.tight_layout()

    out = ROOT / "ruller-paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig_sampled_score_retention_all.{suffix}", dpi=300, bbox_inches="tight")
    csv = out / "fig_sampled_score_retention_all.csv"
    rows = ["task," + ",".join(LEVELS)]
    rows.extend(task + "," + ",".join(f"{v:.6f}" for v in values) for task, values in curves.items())
    csv.write_text("\n".join(rows) + "\n")
    print(out / "fig_sampled_score_retention_all.png")


if __name__ == "__main__":
    main()

