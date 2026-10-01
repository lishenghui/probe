#!/usr/bin/env python3
"""Why is the expand-add epilogue 2.6x off its roofline?

It reads y and writes out, so its floor is 2*M*N elements of traffic.  On the
block sweep it lands near 1.15 TB/s on a GPU that copies at ~3 TB/s, and it is
the dominant sidecar cost for the projections that cannot be grouped.  This
isolates the candidate causes: the strided y that the concat path hands it,
the tile config, and the rank-loop tail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton

from loraforge_kernels.fused_linear import _expand_add_kernel, expand_add, pad_rank

SHAPES = [(32760, 1536), (32760, 8960), (32760, 5120)]
RANKS = [256, 56, 16]


def time_fn(fn, warmup=8, rounds=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record(); fn(); e.record(); e.synchronize()
        best = min(best, b.elapsed_time(e))
    return best


def run_config(y, z, b, out, bm, bn, br, stages, warps):
    m, n = y.shape
    r = z.shape[1]
    _expand_add_kernel.fn[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        y, z, b, out, m, n, r,
        y.stride(0), y.stride(1), z.stride(0), z.stride(1),
        b.stride(0), b.stride(1), out.stride(0), out.stride(1),
        1.0, BM=bm, BN=bn, BR=br, PRECISION="tf32",
        num_stages=stages, num_warps=warps,
    )


CONFIGS = [(128, 128, 3, 8), (64, 256, 3, 8), (256, 64, 3, 8), (128, 64, 4, 4),
           (64, 64, 4, 4), (256, 128, 3, 8), (128, 256, 3, 8), (512, 64, 3, 8),
           (256, 256, 3, 8), (64, 128, 4, 4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/epilogue.json"))
    args = parser.parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    rows = []

    for m, n in SHAPES:
        y_c = torch.randn(m, n, device=dev, dtype=dtype)
        # A pure copy is the floor: same 2*M*N of traffic, no arithmetic.
        dst = torch.empty_like(y_c)
        copy_ms = time_fn(lambda: dst.copy_(y_c))
        gb = 2 * m * n * 2 / 1e9
        print(f"\n=== M={m} N={n} | pure copy {copy_ms*1000:7.1f} us = {gb/copy_ms*1e3:6.0f} GB/s (floor)")
        for r in RANKS:
            z_c = torch.randn(m, r, device=dev, dtype=dtype)
            bmat = torch.randn(n, r, device=dev, dtype=dtype) / r**0.5
            # Strided y, as the concat path produces it.
            wide = torch.randn(m, n + pad_rank(r), device=dev, dtype=dtype)
            y_s = wide[:, :n]
            z_s = wide[:, n : n + r]
            auto_c = time_fn(lambda: expand_add(y_c, z_c, bmat, 1.0))
            auto_s = time_fn(lambda: expand_add(y_s, z_s, bmat, 1.0))
            print(f"  r={r:4d} autotuned contiguous {auto_c*1000:7.1f} us "
                  f"({gb/auto_c*1e3:5.0f} GB/s, {copy_ms/auto_c:4.2f}x of floor) | "
                  f"strided {auto_s*1000:7.1f} us ({gb/auto_s*1e3:5.0f} GB/s)")
            out = torch.empty_like(y_c)
            best = None
            for bm, bn, stages, warps in CONFIGS:
                br = min(pad_rank(r), 64)
                try:
                    ms = time_fn(lambda: run_config(y_c, z_c, bmat, out, bm, bn, br, stages, warps))
                except Exception:
                    continue
                if best is None or ms < best[0]:
                    best = (ms, bm, bn, stages, warps)
            if best:
                ms, bm, bn, stages, warps = best
                print(f"        best explicit config BM={bm} BN={bn} s={stages} w={warps}: "
                      f"{ms*1000:7.1f} us ({gb/ms*1e3:5.0f} GB/s, {copy_ms/ms:4.2f}x of floor)")
            rows.append({"m": m, "n": n, "rank": r, "copy_us": copy_ms * 1000,
                         "auto_contig_us": auto_c * 1000, "auto_strided_us": auto_s * 1000,
                         "best_us": (best[0] * 1000) if best else None,
                         "best_config": best[1:] if best else None})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
