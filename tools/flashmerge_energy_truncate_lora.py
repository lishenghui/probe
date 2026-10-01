#!/usr/bin/env python3
"""FraQ-compress a LoRA to per-module spectral-energy targets."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from analyze_lora_fraq_spectrum import load_flash_extension
from flashmerge_truncate_lora import pair_keys


def retained_ranks(singular: torch.Tensor, threshold: float, multiple: int = 1) -> torch.Tensor:
    """Smallest rank per module reaching `threshold` of the spectral energy.

    `multiple` rounds that rank up.  This is close to free and it matters a lot:
    cuBLAS drops off its tensor-core kernels when a GEMM dimension is not a
    multiple of 8, and the sidecar's rank *is* that dimension.  Measured on a
    Wan attention layer (M=32760, K=N=1536, bf16, GH200), shrink+expand:

        rank 256 (16|r) 156.6 us      rank 241 (unaligned) 603.9 us
        rank   8  (8|r) 129.9 us      rank  11 (unaligned) 226.2 us

    A rank-241 sidecar has 6% fewer flops than rank 256 and costs 3.9x more.
    Rounding up keeps *more* singular values, so accuracy only improves.
    """
    cumulative = singular.square().cumsum(dim=1) / singular.square().sum(dim=1, keepdim=True)
    ranks = (cumulative < threshold).sum(dim=1).add(1)
    if multiple > 1:
        ranks = ranks.add(multiple - 1).div(multiple, rounding_mode="floor").mul(multiple)
    return ranks.clamp_max(singular.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--energy-thresholds", type=float, nargs="+", default=[0.9, 0.8, 0.5])
    parser.add_argument(
        "--rank-multiple", type=int, default=8,
        help="round every retained rank up to this multiple; 1 disables. "
             "Alignment is what keeps the compressed sidecar on cuBLAS's "
             "tensor-core path, and it only adds singular values.",
    )
    parser.add_argument("--rows-per-leaf", type=int, default=256)
    parser.add_argument("--threads-per-block", type=int, default=256)
    parser.add_argument(
        "--flash-kernel",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "FlashTSQR/kernels/tsqr_full.cu",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlashMerge/FraQ compression requires CUDA")
    if any(not 0 < value <= 1 for value in args.energy_thresholds):
        raise ValueError("Energy thresholds must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = load_file(args.checkpoint, device="cpu")
    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    pairs = pair_keys(state)
    buckets: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    for item in pairs:
        _, a_key, b_key = item
        buckets[(tuple(state[a_key].shape), tuple(state[b_key].shape))].append(item)

    lora_keys = {key for _, a, b in pairs for key in (a, b)}
    outputs = {
        target: {key: value for key, value in state.items() if key not in lora_keys}
        for target in args.energy_thresholds
    }
    rank_records: dict[float, dict[str, int]] = {target: {} for target in args.energy_thresholds}
    achieved_records: dict[float, list[float]] = {target: [] for target in args.energy_thresholds}

    # The custom leaf kernel needs rows_per_leaf >= rank and is limited by
    # Hopper's dynamic shared memory. Rank 256 therefore uses the same FraQ
    # QR-core-SVD math through batched torch.linalg.qr; smaller ranks retain
    # the FlashTSQR fast path.
    use_flash = any(state[a_key].shape[0] < 192 for _, a_key, _ in pairs)
    extension = load_flash_extension(args.flash_kernel) if use_flash else None
    started = time.perf_counter()
    for bucket_index, items in enumerate(buckets.values(), 1):
        a_keys = [item[1] for item in items]
        b_keys = [item[2] for item in items]
        a_cpu = torch.stack([state[key] for key in a_keys])
        b_cpu = torch.stack([state[key] for key in b_keys])
        original_dtype = a_cpu.dtype
        a = a_cpu.float().cuda()
        b = b_cpu.float().cuda()
        rank0 = a.shape[1]
        rpl_b = max(rank0, min(args.rows_per_leaf, b.shape[1]))
        rpl_a = max(rank0, min(args.rows_per_leaf, a.shape[2]))
        if rank0 >= 192:
            qb, rb = torch.linalg.qr(b, mode="reduced")
            qa, ra = torch.linalg.qr(a.transpose(1, 2), mode="reduced")
        else:
            rb = extension.tsqr_factor(b, rpl_b, args.threads_per_block)
            ra = extension.tsqr_factor(a.transpose(1, 2).contiguous(), rpl_a, args.threads_per_block)
        u, singular, vh = torch.linalg.svd(rb @ ra.transpose(1, 2))

        for target in args.energy_thresholds:
            ranks = retained_ranks(singular, target, args.rank_multiple)
            max_rank = int(ranks.max().item())
            root_s = singular[:, :max_rank].sqrt()
            left = (u[:, :, :max_rank] * root_s[:, None, :]).contiguous()
            right = (vh[:, :max_rank, :].transpose(1, 2) * root_s[:, None, :]).contiguous()
            if rank0 >= 192:
                b2 = (qb @ left).to(original_dtype).cpu()
                a2 = (qa @ right).transpose(1, 2).to(original_dtype).cpu()
            else:
                extension.tsqr_factor(b, rpl_b, args.threads_per_block)
                b2 = extension.tsqr_applyQ(left, args.threads_per_block).to(original_dtype).cpu()
                extension.tsqr_factor(a.transpose(1, 2).contiguous(), rpl_a, args.threads_per_block)
                a2 = extension.tsqr_applyQ(right, args.threads_per_block).transpose(1, 2).to(original_dtype).cpu()
            total_energy = singular.square().sum(dim=1)
            for position, (prefix, a_key, b_key) in enumerate(items):
                kept = int(ranks[position].item())
                outputs[target][a_key] = a2[position, :kept].contiguous()
                outputs[target][b_key] = b2[position, :, :kept].contiguous()
                alpha_key = prefix + ".alpha"
                if alpha_key in outputs[target]:
                    outputs[target][alpha_key] = torch.full_like(outputs[target][alpha_key], kept)
                rank_records[target][prefix] = kept
                achieved = singular[position, :kept].square().sum() / total_energy[position]
                achieved_records[target].append(float(achieved))
        print(
            f"[{bucket_index}/{len(buckets)}] batch={len(items)} "
            f"A={tuple(a.shape[1:])} B={tuple(b.shape[1:])}",
            flush=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    manifest = {
        "source": str(args.checkpoint),
        "modules": len(pairs),
        "buckets": len(buckets),
        "energy_thresholds": args.energy_thresholds,
        "elapsed_seconds": elapsed,
        "backend": "FlashMerge/FraQ",
        "variants": {},
    }
    for target, tensors in outputs.items():
        label = f"e{round(target * 100):02d}"
        ranks = list(rank_records[target].values())
        achieved = achieved_records[target]
        destination = args.output_dir / f"{args.checkpoint.stem}_fraq_{label}.safetensors"
        out_metadata = dict(metadata)
        out_metadata.update(
            {
                "rank": "variable",
                "alpha": "variable-per-module",
                "fraq_energy_threshold": str(target),
                "fraq_backend": "FlashMerge",
                "fraq_rank_multiple": str(args.rank_multiple),
                "fraq_source": args.checkpoint.name,
            }
        )
        save_file(tensors, destination, metadata=out_metadata)
        stats = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "rank_min": min(ranks),
            "rank_mean": sum(ranks) / len(ranks),
            "rank_max": max(ranks),
            "achieved_energy_min": min(achieved),
            "achieved_energy_mean": sum(achieved) / len(achieved),
            "rank_by_module": rank_records[target],
        }
        manifest["variants"][label] = stats
        print(f"saved {destination} ({destination.stat().st_size / 2**20:.1f} MiB)", flush=True)
    threshold_label = "_".join(f"e{round(value * 100):02d}" for value in args.energy_thresholds)
    (args.output_dir / f"{args.checkpoint.stem}_fraq_energy_{threshold_label}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({**manifest, "variants": {k: {x: y for x, y in v.items() if x != "rank_by_module"} for k, v in manifest["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()
