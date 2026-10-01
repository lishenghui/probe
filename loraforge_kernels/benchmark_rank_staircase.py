#!/usr/bin/env python3
"""Measure storage rank separately from the GPU execution shape.

The controlled comparison is ``torch_compact`` versus ``torch_padded8``:
both use the same two cuBLAS GEMMs, while the latter physically zero-pads A/B
to a multiple of eight.  ``compact_row`` and ``compact_tiled`` keep only the
true rank in global memory and mask a power-of-two rank tile inside Triton.

Sweep every rank so alignment discontinuities and dominated operating points
are visible instead of being inferred from a few hand-picked ranks.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from loraforge_kernels.fused_lora import fused_lora, tiled_lora


def ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def median_us(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def adapter_bytes(rank: int, k: int, n: int, element_size: int) -> int:
    return rank * (k + n) * element_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--ranks", type=int, nargs="*", default=list(range(1, 65)))
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--variant-order",
        choices=("forward", "reverse"),
        default="forward",
        help="reverse the per-point measurement order to expose clock/order bias",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kernel_bench/rank-staircase/rank_staircase.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.ranks or min(args.ranks) < 1 or max(args.ranks) > 64:
        parser.error("ranks must be in [1, 64]")

    torch.manual_seed(20260904)
    device, dtype = "cuda", torch.bfloat16
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".jsonl")
    partial.unlink(missing_ok=True)

    print(
        f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"triton={triton.__version__} dtype={dtype} K={args.k} N={args.n}",
        flush=True,
    )

    for m in args.batches:
        x = torch.randn(m, args.k, device=device, dtype=dtype) / args.k**0.5
        y = torch.randn(m, args.n, device=device, dtype=dtype)
        print(f"\nM={m}", flush=True)
        for rank in args.ranks:
            exec8 = ceil_multiple(rank, 8)
            exec_pow2 = triton.next_power_of_2(rank)
            a = torch.randn(rank, args.k, device=device, dtype=dtype) / args.k**0.5
            b = torch.randn(args.n, rank, device=device, dtype=dtype) / rank**0.5
            a8 = torch.zeros(exec8, args.k, device=device, dtype=dtype)
            b8 = torch.zeros(args.n, exec8, device=device, dtype=dtype)
            a8[:rank].copy_(a)
            b8[:, :rank].copy_(b)

            def torch_compact():
                return y + F.linear(F.linear(x, a), b)

            def torch_padded8():
                return y + F.linear(F.linear(x, a8), b8)

            variants = {
                "torch_compact": torch_compact,
                "torch_padded8": torch_padded8,
                "compact_row": lambda: fused_lora(x, a, b, y),
                "compact_tiled": lambda: tiled_lora(x, a, b, y),
            }
            variant_names = list(variants)
            if args.variant_order == "reverse":
                variant_names.reverse()
            reference = torch_compact()
            timings, errors = {}, {}
            for name in variant_names:
                fn = variants[name]
                got = fn()
                errors[name] = float(
                    (got.float() - reference.float()).abs().max()
                    / reference.float().abs().max().clamp_min(1e-6)
                )
                if errors[name] > 5e-2:
                    raise AssertionError(
                        f"{name} failed correctness at M={m}, rank={rank}: "
                        f"relative error {errors[name]:.3e}"
                    )
                timings[name] = median_us(fn, args.warmup, args.repeats)

            compact_bytes = adapter_bytes(rank, args.k, args.n, a.element_size())
            padded_bytes = adapter_bytes(exec8, args.k, args.n, a.element_size())
            row = {
                "m": m,
                "k": args.k,
                "n": args.n,
                "store_rank": rank,
                "exec_rank_8": exec8,
                "exec_rank_pow2": exec_pow2,
                "compact_bytes": compact_bytes,
                "padded8_bytes": padded_bytes,
                "padding_overhead_pct": 100.0 * (padded_bytes / compact_bytes - 1.0),
                "timings_us": timings,
                "relative_errors": errors,
                "padded8_speedup_pct": 100.0 * (
                    timings["torch_compact"] / timings["torch_padded8"] - 1.0
                ),
            }
            rows.append(row)
            with partial.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"r={rank:2d} q8={exec8:2d} q2={exec_pow2:2d} "
                f"compact={timings['torch_compact']:7.2f}us "
                f"pad8={timings['torch_padded8']:7.2f}us "
                f"row={timings['compact_row']:7.2f}us "
                f"tiled={timings['compact_tiled']:7.2f}us "
                f"pad8_gain={row['padded8_speedup_pct']:+6.1f}%",
                flush=True,
            )

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "dtype": str(dtype),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "variant_order": args.variant_order,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
