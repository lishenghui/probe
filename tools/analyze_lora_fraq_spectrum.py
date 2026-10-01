#!/usr/bin/env python3
"""Analyze per-module LoRA spectra with the small-core FraQ decomposition."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


THRESHOLDS = (0.90, 0.95, 0.99, 0.999)


def module_kind(name: str) -> str:
    match = re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)(?:\.|$)", name)
    if match:
        return match.group(1)
    match = re.search(r"\.(cross_attn|self_attn)\.([^.]+)$", name)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    match = re.search(r"\.ffn\.([^.]+)$", name)
    if match:
        return f"ffn.{match.group(1)}"
    parts = name.split(".")
    if parts[-1].isdigit():
        # Preserve enough path context for Sequential modules such as
        # text_embedding.0 and img_emb.proj.1; a bare "0" is ambiguous.
        return ".".join(parts[-3:] if parts[-2] == "proj" else parts[-2:])
    return parts[-1]


def energy_rank(s: torch.Tensor, threshold: float) -> int:
    cumulative = torch.cumsum(s.square(), dim=0) / s.square().sum()
    return int(torch.searchsorted(cumulative, threshold).item() + 1)


def load_pairs(checkpoint: Path) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(checkpoint, device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    pairs = []
    for key, a in state.items():
        if key.endswith(".lora_A.weight"):
            prefix = key[: -len(".lora_A.weight")]
            b_key = prefix + ".lora_B.weight"
        elif key.endswith(".lora_down.weight"):
            prefix = key[: -len(".lora_down.weight")]
            b_key = prefix + ".lora_up.weight"
        else:
            continue
        if b_key not in state:
            raise KeyError(f"Missing matching tensor: {b_key}")
        pairs.append((prefix, a, state[b_key]))
    if not pairs:
        raise ValueError(f"No lora_A/lora_B pairs found in {checkpoint}")
    return sorted(pairs)


def spectrum(a: torch.Tensor, b: torch.Tensor, device: torch.device) -> torch.Tensor:
    # B A = Q_B (R_B R_A^T) Q_A^T, so only the rank-by-rank core needs an SVD.
    a = a.to(device=device, dtype=torch.float32)
    b = b.to(device=device, dtype=torch.float32)
    _, rb = torch.linalg.qr(b, mode="reduced")
    _, ra = torch.linalg.qr(a.T, mode="reduced")
    return torch.linalg.svdvals(rb @ ra.T).cpu()


def load_flash_extension(kernel: Path):
    from torch.utils.cpp_extension import load_inline

    cpp = (
        "torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);\n"
        "torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb);"
    )
    return load_inline(
        name="flashmerge_fraq",
        cpp_sources=cpp,
        cuda_sources=kernel.read_text(),
        functions=["tsqr_factor", "tsqr_applyQ"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def flash_spectra(
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]],
    extension,
    device: torch.device,
    rows_per_leaf: int,
    threads_per_block: int,
) -> list[torch.Tensor]:
    if device.type != "cuda":
        raise ValueError("The FlashMerge backend requires a CUDA device")
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for index, (_, a, b) in enumerate(pairs):
        buckets[(tuple(a.shape), tuple(b.shape))].append(index)

    output: list[torch.Tensor | None] = [None] * len(pairs)
    print(f"FlashMerge buckets: {len(buckets)} for {len(pairs)} modules", flush=True)
    for bucket_index, indices in enumerate(buckets.values(), 1):
        a = torch.stack([pairs[i][1] for i in indices]).to(device=device, dtype=torch.float32)
        b = torch.stack([pairs[i][2] for i in indices]).to(device=device, dtype=torch.float32)
        rank = a.shape[1]
        rpl_b = max(rank, min(rows_per_leaf, b.shape[1]))
        rpl_a = max(rank, min(rows_per_leaf, a.shape[2]))
        rb = extension.tsqr_factor(b, rpl_b, threads_per_block)
        ra = extension.tsqr_factor(a.transpose(1, 2).contiguous(), rpl_a, threads_per_block)
        values = torch.linalg.svdvals(rb @ ra.transpose(1, 2)).cpu()
        for position, index in enumerate(indices):
            output[index] = values[position]
        print(
            f"[bucket {bucket_index:2d}/{len(buckets)}] batch={len(indices):3d} "
            f"A={tuple(a.shape[1:])} B={tuple(b.shape[1:])}",
            flush=True,
        )
    if any(value is None for value in output):
        raise RuntimeError("FlashMerge did not produce a spectrum for every LoRA module")
    return [value for value in output if value is not None]


def plot_spectra(rows: list[dict], spectra: list[np.ndarray], output: Path) -> None:
    import matplotlib.pyplot as plt

    kinds = sorted({row["kind"] for row in rows})
    colors = dict(zip(kinds, plt.cm.tab10(np.linspace(0, 1, len(kinds)))))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    for row, values in zip(rows, spectra):
        x = np.arange(1, len(values) + 1)
        normalized = values / values[0]
        cumulative = np.cumsum(values**2) / np.sum(values**2)
        color = colors[row["kind"]]
        ax1.semilogy(x, normalized, color=color, alpha=0.10, linewidth=0.7)
        ax2.plot(x, cumulative, color=color, alpha=0.10, linewidth=0.7)

    for kind in kinds:
        selected = [s for row, s in zip(rows, spectra) if row["kind"] == kind]
        median_s = np.median(np.stack([s / s[0] for s in selected]), axis=0)
        median_e = np.median(
            np.stack([np.cumsum(s**2) / np.sum(s**2) for s in selected]), axis=0
        )
        x = np.arange(1, len(median_s) + 1)
        ax1.semilogy(x, median_s, color=colors[kind], linewidth=2, label=kind)
        ax2.plot(x, median_e, color=colors[kind], linewidth=2, label=kind)

    ax1.set(title="LoRA update spectrum", xlabel="Singular-value index", ylabel=r"$\sigma_i/\sigma_1$")
    ax2.set(title="Cumulative Frobenius energy", xlabel="Retained rank", ylabel=r"$\sum_{i\leq k}\sigma_i^2/\sum_i\sigma_i^2$")
    for level in THRESHOLDS:
        ax2.axhline(level, color="gray", linestyle="--", linewidth=0.6)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1, max(len(s) for s in spectra))
    ax1.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("torch", "flash"), default="torch")
    parser.add_argument(
        "--flash-kernel",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "FlashTSQR/kernels/tsqr_full.cu",
    )
    parser.add_argument("--rows-per-leaf", type=int, default=128)
    parser.add_argument("--threads-per-block", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    pairs = load_pairs(args.checkpoint)
    started = time.perf_counter()
    if args.backend == "flash":
        extension = load_flash_extension(args.flash_kernel)
        computed_spectra = flash_spectra(
            pairs, extension, device, args.rows_per_leaf, args.threads_per_block
        )
    else:
        computed_spectra = [spectrum(a, b, device) for _, a, b in pairs]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    rows, all_spectra = [], []
    for index, ((name, a, b), s) in enumerate(zip(pairs, computed_spectra), 1):
        ranks = {f"rank_energy_{t:g}": energy_rank(s, t) for t in THRESHOLDS}
        row = {
            "module": name,
            "kind": module_kind(name),
            "input_dim": a.shape[1],
            "output_dim": b.shape[0],
            "original_rank": a.shape[0],
            **ranks,
        }
        rows.append(row)
        all_spectra.append(s.numpy())
        print(f"[{index:3d}/{len(pairs)}] {name}: r95={ranks['rank_energy_0.95']} r99={ranks['rank_energy_0.99']}", flush=True)

    csv_path = args.output_dir / "fraq_spectrum_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "fraq_spectra.npz",
        module=np.asarray([row["module"] for row in rows]),
        singular_values=np.stack(all_spectra),
    )
    try:
        plot_spectra(rows, all_spectra, args.output_dir / "fraq_spectrum_decay.png")
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print("matplotlib is unavailable; skipped PNG generation", flush=True)

    summary = {
        "modules": len(rows),
        "original_rank": sorted({r["original_rank"] for r in rows}),
        "backend": args.backend,
        "elapsed_seconds_including_extension_load": elapsed,
    }
    for threshold in THRESHOLDS:
        key = f"rank_energy_{threshold:g}"
        values = np.asarray([row[key] for row in rows])
        summary[key] = {
            "min": int(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "max": int(values.max()),
        }
    (args.output_dir / "fraq_spectrum_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
