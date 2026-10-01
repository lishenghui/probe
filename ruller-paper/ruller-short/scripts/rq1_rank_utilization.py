#!/usr/bin/env python3
"""Run the RQ1 rank-utilization audit over rank-32 and rank-64 censuses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THRESHOLDS = (0.90, 0.95, 0.99)


def load_census(root: Path, nominal_rank: int) -> dict:
    artifact = root / f"top100_rank{nominal_rank}_dominated"
    with (artifact / "repos.csv").open(newline="") as handle:
        repos = list(csv.DictReader(handle))
    with (artifact / "layers.csv").open(newline="") as handle:
        layers = list(csv.DictReader(handle))
    spectra = np.load(artifact / "spectra.npz")["singular_values"]
    if len(repos) != 100 or len(layers) != len(spectra):
        raise ValueError(f"Inconsistent rank-{nominal_rank} census")

    values = np.asarray(spectra, dtype=np.float64)
    energy = np.square(values)
    total = energy.sum(axis=1)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(total) & (total > 0)
    cumulative = np.cumsum(energy[valid], axis=1) / total[valid, None]
    retained = {
        threshold: np.argmax(cumulative >= threshold, axis=1) + 1
        for threshold in THRESHOLDS
    }
    modalities = {row["repo_id"]: row.get("modality", "Unlabeled") for row in repos}
    return {
        "rank": nominal_rank,
        "repos": repos,
        "repo_ids": np.asarray([row["repo_id"] for row in layers], dtype=object),
        "modalities": modalities,
        "values": values,
        "energy": energy,
        "total": total,
        "valid": valid,
        "retained": retained,
    }


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
    }


def bootstrap_repo_median(
    repo_values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(repo_values), size=(samples, len(repo_values)))
    estimates = np.median(repo_values[draws], axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def analyze(censuses: list[dict], bootstrap_samples: int, seed: int) -> tuple[dict, list[dict]]:
    per_repo_layers: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    repo_meta: dict[str, dict] = {}
    all_rur: dict[float, list[np.ndarray]] = defaultdict(list)
    rank_rur: dict[int, dict[float, np.ndarray]] = {}

    total_layers = 0
    valid_layers = 0
    invalid_layers = 0
    for census in censuses:
        rank = census["rank"]
        valid = census["valid"]
        total_layers += len(valid)
        valid_layers += int(valid.sum())
        invalid_layers += int((~valid).sum())
        valid_repos = census["repo_ids"][valid]
        rank_rur[rank] = {}
        for threshold in THRESHOLDS:
            rur = census["retained"][threshold] / rank
            all_rur[threshold].append(rur)
            rank_rur[rank][threshold] = rur
            for repo_id, value in zip(valid_repos, rur):
                per_repo_layers[str(repo_id)][threshold].append(float(value))
        for row in census["repos"]:
            repo_id = row["repo_id"]
            repo_meta[repo_id] = {
                "nominal_rank": rank,
                "modality": census["modalities"][repo_id],
                "downloads": int(row["downloads"]),
            }

    repository_rows = []
    for repo_id, meta in repo_meta.items():
        row = {"repo_id": repo_id, **meta}
        for threshold in THRESHOLDS:
            values = np.asarray(per_repo_layers[repo_id][threshold])
            if not len(values):
                raise ValueError(f"Repository has no valid-energy layer: {repo_id}")
            suffix = str(int(threshold * 100))
            row[f"valid_layers_r{suffix}"] = len(values)
            row[f"median_rur_{suffix}"] = float(np.median(values))
            row[f"median_removable_{suffix}"] = float(1 - np.median(values))
        repository_rows.append(row)

    summary: dict = {
        "repositories": len(repo_meta),
        "matched_layers": total_layers,
        "valid_energy_layers": valid_layers,
        "excluded_zero_or_nonfinite_layers": invalid_layers,
        "bootstrap": {
            "unit": "repository",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "thresholds": {},
        "by_nominal_rank": {},
        "by_modality": {},
    }
    for threshold in THRESHOLDS:
        suffix = str(int(threshold * 100))
        layer_values = np.concatenate(all_rur[threshold])
        repo_values = np.asarray([row[f"median_rur_{suffix}"] for row in repository_rows])
        ci_low, ci_high = bootstrap_repo_median(
            repo_values, samples=bootstrap_samples, seed=seed + int(threshold * 100)
        )
        summary["thresholds"][suffix] = {
            "layer_weighted_rur": percentile_summary(layer_values),
            "layer_weighted_removable_median": float(1 - np.median(layer_values)),
            "layers_with_any_removable_rank_fraction": float(np.mean(layer_values < 1)),
            "layers_with_at_least_half_removable_fraction": float(np.mean(layer_values <= 0.5)),
            "repository_balanced_rur": percentile_summary(repo_values),
            "repository_balanced_median_95ci": [ci_low, ci_high],
            "repository_balanced_removable_median": float(1 - np.median(repo_values)),
            "repositories_with_any_median_removable_rank_fraction": float(
                np.mean(repo_values < 1)
            ),
            "repositories_with_at_least_half_median_removable_fraction": float(
                np.mean(repo_values <= 0.5)
            ),
        }

    for rank, by_threshold in rank_rur.items():
        summary["by_nominal_rank"][str(rank)] = {
            str(int(threshold * 100)): {
                "layers": len(by_threshold[threshold]),
                "rur": percentile_summary(by_threshold[threshold]),
            }
            for threshold in THRESHOLDS
        }

    modalities = sorted({row["modality"] for row in repository_rows})
    for modality in modalities:
        selected = [row for row in repository_rows if row["modality"] == modality]
        summary["by_modality"][modality] = {
            "repositories": len(selected),
            **{
                str(int(threshold * 100)): {
                    "repository_balanced_median_rur": float(
                        np.median([row[f"median_rur_{int(threshold * 100)}"] for row in selected])
                    )
                }
                for threshold in THRESHOLDS
            },
        }
    return summary, repository_rows


def make_figure(censuses: list[dict], summary: dict, output: Path) -> None:
    colors = {0.90: "#15966A", 0.95: "#E12D2D", 0.99: "#2864DC"}
    valid_energy = []
    rur_values: dict[float, list[np.ndarray]] = defaultdict(list)
    for census in censuses:
        valid = census["valid"]
        normalized = census["energy"][valid] / census["total"][valid, None]
        valid_energy.append(normalized)
        for threshold in THRESHOLDS:
            rur_values[threshold].append(census["retained"][threshold] / census["rank"])

    first_directions = 12
    energy = np.concatenate([x[:, :first_directions] for x in valid_energy], axis=0)
    median_share = np.median(energy, axis=0)

    # Draw at the final half-column width.  Keep the two panels side by side,
    # but use compact labels and spacing so the source is never down-scaled
    # from a full-width canvas.
    plt.rcParams.update(
        {
            "font.size": 6.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.55), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03)
    ax = axes[0]
    indices = np.arange(1, first_directions + 1)
    ax.bar(indices, 100 * median_share, color="#578BE8", edgecolor="#2864CC")
    ax.set(
        xlabel="Singular direction $k$",
        ylabel="Energy share",
        title="(a) Energy",
        xticks=(1, 4, 8, 12),
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    for threshold in THRESHOLDS:
        rur = np.concatenate(rur_values[threshold])
        removable = np.sort(1 - rur)
        cdf = np.arange(1, len(removable) + 1) / len(removable)
        median = summary["thresholds"][str(int(threshold * 100))][
            "layer_weighted_removable_median"
        ]
        ax.step(
            removable,
            cdf,
            where="post",
            color=colors[threshold],
            linewidth=2.2,
            label=rf"$\tau={threshold:.2f}$ ({100 * median:.1f}%)",
        )
        ax.axvline(median, color=colors[threshold], linestyle=":", alpha=0.65)
    ax.axhline(0.5, color="#61728A", linestyle=":", linewidth=1.5)
    ax.set(
        xlabel=r"Removable fraction $1-r_\tau/r$",
        ylabel="Layer CDF",
        title="(b) Removable rank",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    ax.xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    ax.set_xticks((0, 0.5, 1.0))
    ax.set_yticks((0, 0.5, 1.0))
    ax.tick_params(axis="y", direction="in", length=2.2, pad=1.2)
    ax.set_ylabel("Layer CDF", labelpad=0.8)
    ax.grid(alpha=0.2)
    ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        handlelength=1.1,
        borderpad=0.18,
        labelspacing=0.1,
        handletextpad=0.35,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_repository_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure", type=Path, default=Path("figures/rq1_rank_utilization_200repos.png"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    censuses = [load_census(args.artifacts_root, rank) for rank in (32, 64)]
    summary, repository_rows = analyze(censuses, args.bootstrap_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rq1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_repository_csv(args.output_dir / "rq1_repository_summary.csv", repository_rows)
    make_figure(censuses, summary, args.figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
