#!/usr/bin/env python3
"""Measure scale-relative numerical rank deficiency in the 200-repository audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOLERANCES = (1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
TABLE_TOLERANCES = (1e-12, 1e-8, 1e-6)


def load_census(root: Path, nominal_rank: int) -> dict:
    artifact = root / f"top100_rank{nominal_rank}_dominated"
    with (artifact / "repos.csv").open(newline="") as handle:
        repos = list(csv.DictReader(handle))
    with (artifact / "layers.csv").open(newline="") as handle:
        layers = list(csv.DictReader(handle))
    values = np.asarray(
        np.load(artifact / "spectra.npz")["singular_values"], dtype=np.float64
    )
    if len(repos) != 100 or len(layers) != len(values):
        raise ValueError(f"Inconsistent rank-{nominal_rank} census")
    energy = np.square(values)
    total = energy.sum(axis=1)
    valid = (
        np.isfinite(values).all(axis=1)
        & np.isfinite(total)
        & (total > 0)
        & (values[:, 0] > 0)
    )
    return {
        "rank": nominal_rank,
        "repos": repos,
        "repo_ids": np.asarray([row["repo_id"] for row in layers], dtype=object),
        "values": values[valid],
        "total": total[valid],
        "repo_ids_valid": np.asarray([row["repo_id"] for row in layers], dtype=object)[valid],
        "layers": len(values),
        "valid_layers": int(valid.sum()),
    }


def bootstrap_median(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    medians = np.median(values[draws], axis=1)
    return [float(x) for x in np.quantile(medians, (0.025, 0.975))]


def analyze(censuses: list[dict], samples: int, seed: int) -> tuple[dict, list[dict]]:
    repo_metrics: dict[str, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    repo_rank: dict[str, int] = {}
    layer_metrics: dict[float, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for census in censuses:
        rank = census["rank"]
        relative = census["values"] / census["values"][:, :1]
        for row in census["repos"]:
            repo_rank[row["repo_id"]] = rank
        for tolerance in TOLERANCES:
            keep = relative > tolerance
            retained = keep.sum(axis=1)
            retained_fraction = retained / rank
            removed_fraction = 1.0 - retained_fraction
            discarded_energy = np.where(keep, 0.0, np.square(census["values"])).sum(axis=1)
            relative_error = np.sqrt(discarded_energy / census["total"])
            layer_metrics[tolerance]["retained_fraction"].append(retained_fraction)
            layer_metrics[tolerance]["removed_fraction"].append(removed_fraction)
            layer_metrics[tolerance]["relative_error"].append(relative_error)
            for repo_id, retained_value, removed_value, error_value in zip(
                census["repo_ids_valid"], retained_fraction, removed_fraction, relative_error
            ):
                metrics = repo_metrics[str(repo_id)][tolerance]
                metrics["retained_fraction"].append(float(retained_value))
                metrics["removed_fraction"].append(float(removed_value))
                metrics["relative_error"].append(float(error_value))

    repo_rows = []
    for repo_id in sorted(repo_rank):
        row: dict[str, object] = {"repo_id": repo_id, "nominal_rank": repo_rank[repo_id]}
        for tolerance in TOLERANCES:
            suffix = f"{tolerance:.0e}"
            for metric in ("retained_fraction", "removed_fraction", "relative_error"):
                row[f"median_{metric}_{suffix}"] = float(
                    np.median(repo_metrics[repo_id][tolerance][metric])
                )
        repo_rows.append(row)

    summary: dict[str, object] = {
        "repositories": len(repo_rows),
        "matched_layers": sum(c["layers"] for c in censuses),
        "valid_nonzero_layers": sum(c["valid_layers"] for c in censuses),
        "definition": "r_num(epsilon) = count(sigma_i / sigma_1 > epsilon)",
        "primary_tolerance": 1e-6,
        "float32_machine_epsilon": float(np.finfo(np.float32).eps),
        "bootstrap": {"unit": "repository", "samples": samples, "seed": seed},
        "tolerances": {},
    }
    for index, tolerance in enumerate(TOLERANCES):
        suffix = f"{tolerance:.0e}"
        entry = {}
        for metric in ("retained_fraction", "removed_fraction", "relative_error"):
            layers = np.concatenate(layer_metrics[tolerance][metric])
            repos = np.asarray([row[f"median_{metric}_{suffix}"] for row in repo_rows])
            entry[metric] = {
                "layer_weighted_median": float(np.median(layers)),
                "repository_balanced_median": float(np.median(repos)),
                "repository_bootstrap_median_95ci": bootstrap_median(
                    repos, samples, seed + 100 * index
                ),
            }
        repo_removed = np.asarray(
            [row[f"median_removed_fraction_{suffix}"] for row in repo_rows]
        )
        layer_removed = np.concatenate(layer_metrics[tolerance]["removed_fraction"])
        layer_error = np.concatenate(layer_metrics[tolerance]["relative_error"])
        affected = layer_removed > 0
        entry["layers_with_any_rank_removed_fraction"] = float(np.mean(affected))
        entry["affected_layers"] = int(affected.sum())
        entry["affected_layer_removed_fraction_median"] = (
            float(np.median(layer_removed[affected])) if affected.any() else 0.0
        )
        entry["affected_layer_relative_error_median"] = (
            float(np.median(layer_error[affected])) if affected.any() else 0.0
        )
        entry["repositories_with_any_median_rank_removed_fraction"] = float(
            np.mean(repo_removed > 0)
        )
        summary["tolerances"][suffix] = entry
    return summary, repo_rows


def make_figure(summary: dict, output: Path) -> None:
    tolerances = np.asarray(TOLERANCES)
    entries = [summary["tolerances"][f"{value:.0e}"] for value in tolerances]
    layer_prevalence = 100 * np.asarray(
        [entry["layers_with_any_rank_removed_fraction"] for entry in entries]
    )
    repo_prevalence = 100 * np.asarray(
        [entry["repositories_with_any_median_rank_removed_fraction"] for entry in entries]
    )
    affected_removed = 100 * np.asarray(
        [entry["affected_layer_removed_fraction_median"] for entry in entries]
    )
    affected_error = 100 * np.asarray(
        [entry["affected_layer_relative_error_median"] for entry in entries]
    )

    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.035, hspace=0.02)
    ax = axes[0]
    ax.plot(tolerances, layer_prevalence, marker="o", color="#2864DC", linewidth=2, label="Layers")
    ax.plot(tolerances, repo_prevalence, marker="s", color="#15966A", linewidth=2, label="Repositories")
    ax.set(xscale="log", yscale="log", xlabel=r"Relative tolerance $\epsilon$", ylabel="Deficient (\%)", title="Prevalence of deficient updates")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    ax.set_xticks((1e-10, 1e-6, 1e-2))

    ax = axes[1]
    ax.plot(tolerances, affected_removed, marker="o", color="#2864DC", linewidth=2, label="Rank removed")
    ax.plot(tolerances, affected_error, marker="s", color="#E12D2D", linewidth=2, label=r"$\|\cdot\|_F$ error")
    ax.set(xscale="log", xlabel=r"Relative tolerance $\epsilon$", ylabel="Median (\%)", title="Magnitude when deficiency occurs", ylim=(-2, 102))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    ax.set_xticks((1e-10, 1e-6, 1e-2))
    for ax in axes:
        ax.axvline(1e-6, color="#15966A", linestyle=":", linewidth=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure", type=Path, default=Path("figures/rq2_numerical_redundancy.png"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    censuses = [load_census(args.artifacts_root, rank) for rank in (32, 64)]
    summary, rows = analyze(censuses, args.bootstrap_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rq2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output_dir / "rq2_repository_summary.csv", rows)
    make_figure(summary, args.figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
