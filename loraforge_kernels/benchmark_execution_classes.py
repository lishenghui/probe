#!/usr/bin/env python3
"""Validate that many compact physical ranks share few executable classes."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from loraforge_kernels.fused_lora import (
    _classed_lora_output_tile_kernel,
    _classed_lora_row_kernel,
    _fused_lora_output_tile_kernel,
    _fused_lora_row_kernel,
    classed_lora,
    classed_tiled_lora,
    fused_lora,
    tiled_lora,
)


def ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def cache_entries(kernel) -> int:
    """Count in-process compiled specializations across active devices."""
    return sum(len(cache_tuple[0]) for cache_tuple in kernel.device_caches.values())


def paired_timings(functions, warmup: int, repeats: int, seed: int) -> dict[str, float]:
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    rng = random.Random(seed)
    names = list(functions)
    for _ in range(repeats):
        rng.shuffle(names)
        for name in names:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000.0)
    return {name: statistics.median(values) for name, values in samples.items()}


def compile_audit(k: int, n: int, dtype: torch.dtype) -> dict:
    """Compile ranks 1..64 and record when each JIT cache grows."""
    kernels = {
        "exact_row": _fused_lora_row_kernel,
        "exact_tiled": _fused_lora_output_tile_kernel,
        "classed_row": _classed_lora_row_kernel,
        "classed_tiled": _classed_lora_output_tile_kernel,
    }
    before = {name: cache_entries(kernel) for name, kernel in kernels.items()}
    growth = {name: [] for name in kernels}
    x = torch.randn(1, k, device="cuda", dtype=dtype)
    y = torch.randn(1, n, device="cuda", dtype=dtype)
    started = time.perf_counter()
    for rank in range(1, 65):
        a = torch.randn(rank, k, device="cuda", dtype=dtype)
        b = torch.randn(n, rank, device="cuda", dtype=dtype)
        previous = {name: cache_entries(kernel) for name, kernel in kernels.items()}
        fused_lora(x, a, b, y)
        tiled_lora(x, a, b, y)
        classed_lora(x, a, b, y)
        classed_tiled_lora(x, a, b, y)
        torch.cuda.synchronize()
        for name, kernel in kernels.items():
            now = cache_entries(kernel)
            if now != previous[name]:
                growth[name].append({"rank": rank, "new_entries": now - previous[name]})
    after = {name: cache_entries(kernel) for name, kernel in kernels.items()}
    return {
        "seconds": time.perf_counter() - started,
        "entries_before": before,
        "entries_after": after,
        "entries_added": {name: after[name] - before[name] for name in kernels},
        "growth": growth,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 4, 8, 16, 32, 128, 512])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/kernel_bench/execution-classes/execution_classes.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(20260904)
    dtype = torch.bfloat16
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".jsonl")
    partial.unlink(missing_ok=True)
    print(f"GPU={torch.cuda.get_device_name(0)} K={args.k} N={args.n}", flush=True)

    audit = compile_audit(args.k, args.n, dtype)
    print("compile audit", json.dumps(audit), flush=True)
    rows = []
    for m in args.batches:
        x = torch.randn(m, args.k, device="cuda", dtype=dtype) / args.k**0.5
        y = torch.randn(m, args.n, device="cuda", dtype=dtype)
        print(f"M={m}", flush=True)
        for rank in range(1, 65):
            q8 = ceil_multiple(rank, 8)
            execution_class = triton.next_power_of_2(rank)
            a = torch.randn(rank, args.k, device="cuda", dtype=dtype) / args.k**0.5
            b = torch.randn(args.n, rank, device="cuda", dtype=dtype) / rank**0.5
            a8 = torch.zeros(q8, args.k, device="cuda", dtype=dtype)
            b8 = torch.zeros(args.n, q8, device="cuda", dtype=dtype)
            a8[:rank].copy_(a)
            b8[:, :rank].copy_(b)

            functions = {
                "torch_compact": lambda: y + F.linear(F.linear(x, a), b),
                "torch_padded8": lambda: y + F.linear(F.linear(x, a8), b8),
                "exact_row": lambda: fused_lora(x, a, b, y),
                "classed_row": lambda: classed_lora(x, a, b, y),
                "exact_tiled": lambda: tiled_lora(x, a, b, y),
                "classed_tiled": lambda: classed_tiled_lora(x, a, b, y),
            }
            reference = functions["torch_compact"]()
            errors = {}
            for name, fn in functions.items():
                got = fn()
                errors[name] = float(
                    (got.float() - reference.float()).abs().max()
                    / reference.float().abs().max().clamp_min(1e-6)
                )
                if errors[name] > 5e-2:
                    raise AssertionError(f"{name} M={m} rank={rank} error={errors[name]:.3e}")
            timings = paired_timings(
                functions, args.warmup, args.repeats, seed=20260904 + 1000 * m + rank
            )
            row = {
                "m": m, "k": args.k, "n": args.n,
                "physical_rank": rank, "execution_class": execution_class, "padded8_rank": q8,
                "compact_bytes": rank * (args.k + args.n) * 2,
                "padded8_bytes": q8 * (args.k + args.n) * 2,
                "timings_us": timings, "relative_errors": errors,
            }
            rows.append(row)
            with partial.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"r={rank:2d} c={execution_class:2d} "
                f"exact={timings['exact_tiled']:7.2f}us "
                f"classed={timings['classed_tiled']:7.2f}us",
                flush=True,
            )

    result = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "triton": triton.__version__, "dtype": str(dtype),
        "warmup": args.warmup, "repeats": args.repeats,
        "compile_audit": audit, "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
