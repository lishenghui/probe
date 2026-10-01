"""Preview greedy-vs-sampled retained-utility decay for four LoRA Land adapters."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEVELS = ["e99", "e95", "e90", "e80", "e70", "e50"]
DATA = {
    "CoNLL-PP": {
        "Greedy": [0.99, 0.99, 0.99, 0.98, 0.95, 0.88],
        "Sampled": [1.00, 1.00, 0.97, 0.94, 0.93, 0.75],
    },
    "DBpedia": {
        "Greedy": [0.94, 0.01, 0.08, 0.31, 0.00, 0.00],
        "Sampled": [0.94, 0.07, 0.04, 0.04, 0.00, 0.00],
    },
    "WikiSQL": {
        "Greedy": [1.00, 1.01, 1.02, 1.01, 0.10, -0.01],
        "Sampled": [1.02, 1.00, 1.00, 0.80, 0.16, -0.03],
    },
    "GSM8K": {
        "Greedy": [0.95, 0.97, 0.86, 0.74, -0.29, 0.47],
        "Sampled": None,
    },
}


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)
    x = np.arange(len(LEVELS))
    styles = {
        "Greedy": dict(color="#17365D", marker="o", linestyle="-"),
        "Sampled": dict(color="#D95F02", marker="s", linestyle="--"),
    }

    for ax, (adapter, curves) in zip(axes.flat, DATA.items()):
        ax.axhspan(-0.35, 0, color="#F8E7E7", alpha=0.7, zorder=0)
        ax.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
        ax.axhline(0, color="#777777", linewidth=0.8, zorder=1)
        for decode, values in curves.items():
            if values is None:
                continue
            ax.plot(
                x,
                values,
                label=decode,
                linewidth=2,
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
                **styles[decode],
            )
        if curves["Sampled"] is None:
            ax.text(
                0.5,
                0.13,
                "sampled undefined:\norig. gain $\\approx 0$",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#D95F02",
                fontsize=8.5,
            )
        ax.set_title(adapter, loc="left", fontweight="bold")
        ax.set_xticks(x, LEVELS)
        ax.set_ylim(-0.36, 1.10)
        ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("Retained utility $R$")
    axes[1, 0].set_ylabel("Retained utility $R$")
    axes[1, 0].set_xlabel("Energy threshold (more compression $\\rightarrow$)")
    axes[1, 1].set_xlabel("Energy threshold (more compression $\\rightarrow$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Compression decay depends on the adapter, not the decoding rule", y=1.06, fontweight="bold")
    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig_decode_decay_preview.{suffix}", dpi=220, bbox_inches="tight")
    print(out / "fig_decode_decay_preview.png")


if __name__ == "__main__":
    main()
