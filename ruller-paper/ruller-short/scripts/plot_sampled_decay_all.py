"""Plot sampled-decoding retained utility for all ten LoRA Land tasks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "rq3" / "results"
LEVELS = ["e99", "e95", "e90", "e80", "e70", "e50"]
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


def retained_utility(path: Path) -> list[float]:
    record = json.loads(path.read_text())[0]
    base, original = record["metric_base"], record["metric_orig"]
    gain = original - base
    if abs(gain) < 0.05:
        raise ValueError(f"unstable retained-utility denominator in {path}: {gain}")
    metrics = [record["variants"][level]["metric"] for level in LEVELS]
    return [(metric - base) / gain for metric in metrics]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(LEVELS))
    colors = plt.get_cmap("tab10").colors

    table = {}
    for color, (task, filename) in zip(colors, TASKS.items()):
        values = retained_utility(RESULTS / filename)
        table[task] = values
        ax.plot(
            x,
            values,
            color=color,
            marker="o",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.8,
            label=task,
        )

    ax.axhspan(-0.1, 0, color="#F8E7E7", alpha=0.75, zorder=0)
    ax.axhline(1, color="#666666", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, LEVELS)
    ax.set_ylim(-0.1, 1.10)
    ax.set_ylabel("Retained utility $R$")
    ax.set_xlabel("Energy threshold (more compression $\\rightarrow$)")
    ax.set_title("Sampled decoding reveals adapter-specific failure boundaries", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.55), frameon=False)
    ax.text(
        1.01,
        0.08,
        "GSM8K: undefined\n(adapter gain $\\approx 0$)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#666666",
        fontsize=8.5,
    )
    fig.tight_layout()

    out = ROOT / "ruller-paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig_sampled_decay_all.{suffix}", dpi=240, bbox_inches="tight")

    csv = out / "fig_sampled_decay_all.csv"
    rows = ["task," + ",".join(LEVELS)]
    rows.extend(task + "," + ",".join(f"{value:.6f}" for value in values) for task, values in table.items())
    rows.append("GSM8K," + ",".join(["NA"] * len(LEVELS)))
    csv.write_text("\n".join(rows) + "\n")
    print(out / "fig_sampled_decay_all.png")


if __name__ == "__main__":
    main()
