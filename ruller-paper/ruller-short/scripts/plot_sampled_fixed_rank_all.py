"""Plot sampled task-score retention for fixed retained LoRA ranks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "rq3" / "results"
# Left to right means increasingly aggressive compression.
RANKS = ["k06", "k05", "k04", "k03", "k02", "k01"]
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


def record(filename: str) -> dict:
    return json.loads((RESULTS / filename).read_text())[0]


def ordinary_curve(filename: str) -> list[float]:
    rec = record(filename)
    return [rec["variants"][rank]["metric"] / rec["metric_orig"] for rank in RANKS]


def gsm8k_curve() -> list[float]:
    locations = {
        "k01": 3,
        "k02": 0,
        "k03": 1,
        "k04": 2,
        "k05": 3,
        "k06": 0,
    }
    values = []
    for rank in RANKS:
        rec = record(f"predibase_samp_gsm8k_{locations[rank]}.json")
        values.append(rec["variants"][rank]["metric"] / rec["metric_orig"])
    return values


def main() -> None:
    curves = {task: ordinary_curve(filename) for task, filename in TASKS.items()}
    curves["GSM8K"] = gsm8k_curve()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(RANKS))
    for color, (task, values) in zip(plt.get_cmap("tab10").colors, curves.items()):
        ax.plot(x, values, color=color, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.5,
                linewidth=1.8, label=task)

    ax.axhline(1, color="#666666", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, [rank.replace("k0", "k=") for rank in RANKS])
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel("Task-score retention  $m_{comp}/m_{orig}$")
    ax.set_xlabel("Retained rank (more compression $\\rightarrow$)")
    ax.set_title("Sampled downstream performance at fixed retained rank", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout()

    out = ROOT / "ruller-paper" / "figures"
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig_sampled_fixed_rank_all.{suffix}", dpi=240, bbox_inches="tight")
    csv = out / "fig_sampled_fixed_rank_all.csv"
    rows = ["task," + ",".join(RANKS)]
    rows.extend(task + "," + ",".join(f"{v:.6f}" for v in values) for task, values in curves.items())
    csv.write_text("\n".join(rows) + "\n")
    print(out / "fig_sampled_fixed_rank_all.png")


if __name__ == "__main__":
    main()
