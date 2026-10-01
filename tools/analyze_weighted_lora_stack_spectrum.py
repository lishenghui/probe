#!/usr/bin/env python3
"""Analyze the exact spectrum of a weighted sum of compatible LoRA factors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analyze_lora_fraq_spectrum import module_kind
from flashmerge_weighted_stack_energy import pairs, unwrap


THRESHOLDS = (0.70, 0.80, 0.90, 0.95, 0.99, 0.999)


def energy_rank(values: np.ndarray, threshold: float) -> int:
    cumulative = np.cumsum(values * values) / np.sum(values * values)
    return int(np.searchsorted(cumulative, threshold) + 1)


def plot_spectra(rows: list[dict], spectra: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt

    normalized = spectra / spectra[:, :1]
    cumulative = np.cumsum(spectra**2, axis=1) / np.sum(spectra**2, axis=1, keepdims=True)
    x = np.arange(1, spectra.shape[1] + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    for curve in normalized:
        ax1.semilogy(x, curve, color="tab:blue", alpha=0.055, linewidth=0.65)
    for curve in cumulative:
        ax2.plot(x, curve, color="tab:blue", alpha=0.055, linewidth=0.65)

    for percentile, style, alpha in ((10, "--", 0.8), (50, "-", 1.0), (90, "--", 0.8)):
        ax1.semilogy(
            x, np.percentile(normalized, percentile, axis=0), color="black",
            linestyle=style, linewidth=2 if percentile == 50 else 1.2,
            alpha=alpha, label=f"module p{percentile}",
        )
        ax2.plot(
            x, np.percentile(cumulative, percentile, axis=0), color="black",
            linestyle=style, linewidth=2 if percentile == 50 else 1.2, alpha=alpha,
        )

    ax1.set_title("Weighted 4-MotionLoRA spectrum (168 modules)")
    ax1.set_xlabel("Singular-value index / retained rank")
    ax1.set_ylabel(r"Normalized singular value $\sigma_i/\sigma_1$")
    ax2.set_title("Cumulative Frobenius energy")
    ax2.set_xlabel("Retained rank")
    ax2.set_ylabel(r"$\sum_{i\leq k}\sigma_i^2/\sum_i\sigma_i^2$")
    for level in THRESHOLDS:
        ax2.axhline(level, color="gray", linestyle=":", linewidth=0.7)
    for ax in (ax1, ax2):
        ax.set_xlim(1, spectra.shape[1])
        ax.grid(True, alpha=0.2)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.weight):
        raise ValueError("--checkpoint and --weight counts differ")
    if any(weight <= 0 for weight in args.weight):
        raise ValueError("Weights must be positive")

    states = [unwrap(path) for path in args.checkpoint]
    reference = pairs(states[0])
    reference_keys = [(down, up) for _, down, up in reference]
    for path, state in zip(args.checkpoint[1:], states[1:]):
        if [(down, up) for _, down, up in pairs(state)] != reference_keys:
            raise ValueError(f"LoRA module layout differs: {path}")

    stacked = []
    for prefix, down_key, up_key in reference:
        downs, ups = [], []
        for state, weight in zip(states, args.weight):
            scale = math.sqrt(weight)
            downs.append(state[down_key].float() * scale)
            ups.append(state[up_key].float() * scale)
        stacked.append((prefix, torch.cat(downs, dim=0), torch.cat(ups, dim=1)))

    device = torch.device(args.device)
    buckets: dict[tuple, list[tuple[str, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for item in stacked:
        buckets[(tuple(item[1].shape), tuple(item[2].shape))].append(item)

    spectra_by_name: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for bucket_index, items in enumerate(buckets.values(), 1):
        a = torch.stack([item[1] for item in items]).to(device)
        b = torch.stack([item[2] for item in items]).to(device)
        _, rb = torch.linalg.qr(b, mode="reduced")
        _, ra = torch.linalg.qr(a.transpose(1, 2), mode="reduced")
        values = torch.linalg.svdvals(rb @ ra.transpose(1, 2)).cpu().numpy()
        for item, spectrum in zip(items, values):
            spectra_by_name[item[0]] = spectrum
        print(f"[{bucket_index}/{len(buckets)}] modules={len(items)} rank={a.shape[1]}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    names = [prefix for prefix, _, _ in stacked]
    spectra = np.stack([spectra_by_name[name] for name in names])
    rows = []
    for (name, a, b), values in zip(stacked, spectra):
        row = {
            "module": name,
            "kind": module_kind(name),
            "input_dim": int(a.shape[1]),
            "output_dim": int(b.shape[0]),
            "stacked_rank": int(a.shape[0]),
        }
        row.update({f"rank_energy_{threshold:g}": energy_rank(values, threshold) for threshold in THRESHOLDS})
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fraq_spectrum_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "fraq_spectra.npz",
        module=np.asarray(names), singular_values=spectra,
    )
    plot_spectra(rows, spectra, args.output_dir / "fraq_spectrum_decay.png")

    summary = {
        "operation": "weighted-stack-spectrum",
        "sources": [{"path": str(path), "weight": weight} for path, weight in zip(args.checkpoint, args.weight)],
        "modules": len(rows),
        "stacked_rank": int(spectra.shape[1]),
        "elapsed_seconds": elapsed,
    }
    for threshold in THRESHOLDS:
        key = f"rank_energy_{threshold:g}"
        ranks = np.asarray([row[key] for row in rows])
        summary[key] = {
            "min": int(ranks.min()), "median": float(np.median(ranks)),
            "mean": float(ranks.mean()), "max": int(ranks.max()),
        }
    (args.output_dir / "fraq_spectrum_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
