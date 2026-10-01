#!/usr/bin/env python3
"""Comprehensive Level 2 Benchmark: Split-K, Grouped Projections, and Dual-Stream Overlap."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from loraforge_kernels import (
    DualStreamOverlappedLinear,
    fused_lora,
    grouped_lora,
    splitk_persistent_fused_lora,
)


def time_us(fn, warmup=50, repeats=300):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000)
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "std_us": statistics.stdev(samples),
    }


def main():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16
    k = n = 4096
    ranks = (8, 16, 24, 32, 48, 64)
    batch_sizes = (1, 4, 8)

    results = []

    print(f"Running Level 2 Benchmark on: {torch.cuda.get_device_name(0)}")

    for m in batch_sizes:
        x = torch.randn(m, k, device=device, dtype=dtype)
        w = torch.randn(n, k, device=device, dtype=dtype) / k**0.5

        # 1. Base GEMM baseline
        base_t = time_us(lambda: torch.matmul(x, w.t()))
        print(f"\n[M={m}] Base GEMM Median Latency: {base_t['median_us']:.2f} us")

        for r in ranks:
            a = torch.randn(r, k, device=device, dtype=dtype) / k**0.5
            b = torch.randn(n, r, device=device, dtype=dtype) / max(r, 1)**0.5
            y_base = torch.matmul(x, w.t())

            # 2. Standard PEFT (Sequential)
            peft_t = time_us(lambda: torch.matmul(x, w.t()) + (x @ a.t()) @ b.t())

            # 3. Level 1 Fused LoRA
            fused_l1_t = time_us(lambda: fused_lora(x, a, b, y_base))

            # 4. Level 2 Split-K Fused LoRA
            splitk_t = time_us(lambda: splitk_persistent_fused_lora(x, a, b, y_base, split_k=8))

            # 5. Level 2 Dual-Stream Overlapped Linear
            overlapped_mod = DualStreamOverlappedLinear(k, n, r, dtype=dtype, device=device)
            overlapped_t = time_us(lambda: overlapped_mod(x))

            row = {
                "m": m,
                "r": r,
                "base_us": base_t["median_us"],
                "peft_us": peft_t["median_us"],
                "fused_l1_us": fused_l1_t["median_us"],
                "splitk_us": splitk_t["median_us"],
                "overlapped_us": overlapped_t["median_us"],
                "overlap_hidden_pct": max(0.0, 100.0 * (peft_t["median_us"] - overlapped_t["median_us"]) / (peft_t["median_us"] - base_t["median_us"])),
                "speedup_vs_peft": peft_t["median_us"] / overlapped_t["median_us"],
            }
            results.append(row)
            print(
                f"  M={m:2d}, R={r:2d} | Base: {base_t['median_us']:5.1f}us | PEFT: {peft_t['median_us']:6.1f}us | "
                f"L1 Fused: {fused_l1_t['median_us']:5.1f}us | SplitK: {splitk_t['median_us']:5.1f}us | "
                f"Overlapped: {overlapped_t['median_us']:5.1f}us (Speedup: {row['speedup_vs_peft']:.2f}x, Hidden: {row['overlap_hidden_pct']:.1f}%)"
            )

    # Multi-projection grouping benchmark (QKV, p=3 and Gate-Up, p=2)
    grouped_results = []
    print("\n=== Grouped Projection Level 2 Benchmark ===")
    for p, label in ((2, "gate_up"), (3, "qkv")):
        for m in batch_sizes:
            for r in (8, 16, 24, 32):
                x = torch.randn(m, k, device=device, dtype=dtype)
                aa = torch.randn(p, r, k, device=device, dtype=dtype) / k**0.5
                a_cat = aa.reshape(p * r, k).contiguous()
                bb = torch.randn(p, n, r, device=device, dtype=dtype) / r**0.5
                y = torch.randn(m, p, n, device=device, dtype=dtype)

                def naive_multi():
                    return torch.stack([y[:, i] + (x @ aa[i].t()) @ bb[i].t() for i in range(p)], 1)

                naive_t = time_us(naive_multi)
                grouped_t = time_us(lambda: grouped_lora(x, a_cat, bb, y))

                row = {
                    "kind": label,
                    "p": p,
                    "m": m,
                    "r": r,
                    "effective_rank": p * r,
                    "naive_us": naive_t["median_us"],
                    "grouped_us": grouped_t["median_us"],
                    "speedup": naive_t["median_us"] / grouped_t["median_us"],
                }
                grouped_results.append(row)
                print(f"  {label:8s} (p={p}) | M={m:2d}, R={r:2d} (Eff.R={p*r:2d}) | Naive: {naive_t['median_us']:6.1f}us -> Grouped: {grouped_t['median_us']:5.1f}us | Speedup: {row['speedup']:.2f}x")

    out_path = Path("artifacts/kernel_bench/level2_fusion_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "whole_linear_results": results,
        "grouped_results": grouped_results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved full Level 2 benchmark data to: {out_path}")


if __name__ == "__main__":
    main()
