#!/usr/bin/env python3
"""Truncate every LoRA module in a safetensors adapter with FlashMerge/FraQ."""

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


def pair_keys(state: dict[str, torch.Tensor]):
    pairs = []
    for down_key in state:
        if down_key.endswith(".lora_down.weight"):
            prefix = down_key[: -len(".lora_down.weight")]
            up_key = prefix + ".lora_up.weight"
        elif down_key.endswith(".lora_A.weight"):
            prefix = down_key[: -len(".lora_A.weight")]
            up_key = prefix + ".lora_B.weight"
        else:
            continue
        if up_key not in state:
            raise KeyError(f"Missing matching LoRA tensor: {up_key}")
        pairs.append((prefix, down_key, up_key))
    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[48, 32, 16])
    parser.add_argument("--rows-per-leaf", type=int, default=128)
    parser.add_argument("--threads-per-block", type=int, default=256)
    parser.add_argument(
        "--flash-kernel",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "FlashTSQR/kernels/tsqr_full.cu",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlashMerge truncation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = load_file(args.checkpoint, device="cpu")
    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    pairs = pair_keys(state)
    buckets = defaultdict(list)
    for item in pairs:
        _, down_key, up_key = item
        buckets[(tuple(state[down_key].shape), tuple(state[up_key].shape))].append(item)
    original_ranks = {state[down].shape[0] for _, down, _ in pairs}
    if any(rank <= 0 or rank > min(original_ranks) for rank in args.ranks):
        raise ValueError(f"Requested ranks {args.ranks} incompatible with originals {original_ranks}")

    extension = load_flash_extension(args.flash_kernel)
    outputs = {
        rank: {key: value for key, value in state.items() if ".lora_down.weight" not in key
               and ".lora_up.weight" not in key and ".lora_A.weight" not in key
               and ".lora_B.weight" not in key}
        for rank in args.ranks
    }
    started = time.perf_counter()
    for bucket_index, items in enumerate(buckets.values(), 1):
        down_keys = [item[1] for item in items]
        up_keys = [item[2] for item in items]
        a_cpu = torch.stack([state[key] for key in down_keys])
        b_cpu = torch.stack([state[key] for key in up_keys])
        dtype = a_cpu.dtype
        a = a_cpu.float().cuda()
        b = b_cpu.float().cuda()
        rank0 = a.shape[1]
        rpl_b = max(rank0, min(args.rows_per_leaf, b.shape[1]))
        rpl_a = max(rank0, min(args.rows_per_leaf, a.shape[2]))
        rb = extension.tsqr_factor(b, rpl_b, args.threads_per_block)
        ra = extension.tsqr_factor(a.transpose(1, 2).contiguous(), rpl_a, args.threads_per_block)
        u, singular, vh = torch.linalg.svd(rb @ ra.transpose(1, 2))
        for retained in args.ranks:
            root_s = singular[:, :retained].sqrt()
            left = (u[:, :, :retained] * root_s[:, None, :]).contiguous()
            right = (vh[:, :retained, :].transpose(1, 2) * root_s[:, None, :]).contiguous()
            extension.tsqr_factor(b, rpl_b, args.threads_per_block)
            b2 = extension.tsqr_applyQ(left, args.threads_per_block).to(dtype).cpu().contiguous()
            extension.tsqr_factor(a.transpose(1, 2).contiguous(), rpl_a, args.threads_per_block)
            a2 = extension.tsqr_applyQ(right, args.threads_per_block).transpose(1, 2).to(dtype).cpu().contiguous()
            for position, (down_key, up_key) in enumerate(zip(down_keys, up_keys)):
                outputs[retained][down_key] = a2[position]
                outputs[retained][up_key] = b2[position]
        print(f"[{bucket_index}/{len(buckets)}] batch={len(items)} A={tuple(a.shape[1:])} B={tuple(b.shape[1:])}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    manifest = {"source": str(args.checkpoint), "modules": len(pairs), "buckets": len(buckets),
                "ranks": args.ranks, "elapsed_seconds": elapsed, "backend": "FlashMerge"}
    for rank, tensors in outputs.items():
        destination = args.output_dir / f"{args.checkpoint.stem}_fraq_rank{rank}.safetensors"
        out_metadata = dict(metadata or {})
        out_metadata.update({"fraq_rank": str(rank), "fraq_backend": "FlashMerge",
                             "fraq_source": args.checkpoint.name})
        save_file(tensors, destination, metadata=out_metadata)
        print(f"saved {destination} ({destination.stat().st_size / 2**20:.1f} MiB)", flush=True)
    (args.output_dir / f"{args.checkpoint.stem}_fraq_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
