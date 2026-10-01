#!/usr/bin/env python3
"""Measure how LoRA inference cost scales with rank, per implementation.

The compression story (FraQ e95/e90) only pays off end-to-end if the sidecar's
cost is proportional to the retained rank.  This benchmark reports, for the
layer shapes that dominate the AnimateDiff and Wan runs, the sidecar overhead
over the un-adapted base GEMM for each implementation:

  torch   -- what the runtime sidecar does today: two cuBLAS GEMMs plus `y + a*u`
  hybrid  -- cuBLAS GEMMs plus the fused expand-add epilogue
  concat  -- shrink packed into the weight, one cuBLAS GEMM, fused epilogue
  fused   -- the single Triton kernel that adds no memory traffic at all

Ranks default to the per-module means FraQ actually produces for the AnimateDiff
MotionLoRA stack at e100/e95/e90/e80/e70.

Variants are timed interleaved and reported as minimum-of-N: these shapes are
short enough that clock drift across a sequential sweep is larger than the
effect being measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from loraforge_kernels.fused_linear import (
    _concat,
    _hybrid,
    _torch_naive,
    augment_bias,
    augment_weight,
    eligible,
    fused_gemm,
    pad_rank,
)

# (label, M, K, N) -- rows are batch*tokens as seen by the Linear layer.
SHAPES = [
    ("animatediff/mm-320", 131072, 320, 320),
    ("animatediff/mm-640", 32768, 640, 640),
    ("animatediff/mm-1280", 8192, 1280, 1280),
    ("wan2.1/attn-1536", 32760, 1536, 1536),
    ("wan2.1/ffn-8960", 32760, 1536, 8960),
    ("llm/prefill-4096", 4096, 4096, 4096),
]

# FraQ rank_mean for the AnimateDiff MotionLoRA stack, per energy target.
RANKS = [("e100", 256), ("e95", 40), ("e90", 28), ("e80", 17), ("e70", 11)]

ORDER = ("torch", "hybrid", "concat", "fused")


def time_all(fns: dict, warmup=10, rounds=30):
    """Interleave the candidates so clock drift hits every variant equally."""
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    best = {name: float("inf") for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            fn()
            end.record()
            end.synchronize()
            best[name] = min(best[name], begin.elapsed_time(end))
    return best


def rel_err(candidate, reference):
    denom = reference.abs().max().clamp_min(1e-6)
    return float((candidate.float() - reference).abs().max() / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/rank_scaling.json"))
    parser.add_argument("--shape", action="append", default=None, help="restrict to shape labels")
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda"
    torch.manual_seed(0)
    shapes = [s for s in SHAPES if args.shape is None or s[0] in args.shape]

    rows = []
    print(f"GPU: {torch.cuda.get_device_name(0)}  dtype={args.dtype}")
    for label, m, k, n in shapes:
        x = torch.randn(m, k, device=device, dtype=dtype) / k**0.5
        w = torch.randn(n, k, device=device, dtype=dtype) / k**0.5
        bias = torch.randn(n, device=device, dtype=dtype)

        # Two references: cuBLAS on the un-adapted layer, and the fused kernel
        # with the sidecar switched off.  The gap between them is how much of
        # the fused path's cost is GEMM quality rather than sidecar work.
        refs = {"cublas_base": lambda: torch.nn.functional.linear(x, w, bias)}
        if dtype != torch.float32:
            refs["triton_base"] = lambda: fused_gemm(x, w, bias, None, None, 1.0)
        base_t = time_all(refs, rounds=args.rounds)
        triton_base = base_t.get("triton_base")
        note = f" | triton base {triton_base*1000:7.1f} us" if triton_base else ""
        print(f"\n=== {label}  M={m} K={k} N={n} | cuBLAS base {base_t['cublas_base']*1000:8.1f} us{note}")
        print(f"{'variant':8s} {'rank':>5s} {'pad':>4s} {'total us':>10s} {'lora us':>9s} "
              f"{'vs base':>8s} {'rel err':>9s}")
        if triton_base:
            rows.append({"shape": label, "m": m, "k": k, "n": n, "variant": "triton_base",
                         "rank": 0, "total_us": triton_base * 1000})

        for tag, r in RANKS:
            a = torch.randn(r, k, device=device, dtype=dtype) / k**0.5
            b = torch.randn(n, r, device=device, dtype=dtype) / r**0.5
            scale = 1.0
            w_aug = augment_weight(w, a)
            b_aug = augment_bias(bias, r)

            impls = {}
            # The un-adapted base is timed inside the same interleaved group as
            # the variants: on these short kernels clock drift between two
            # separate timing calls is larger than the sidecar being measured.
            impls["base"] = lambda: torch.nn.functional.linear(x, w, bias)
            impls["torch"] = lambda: _torch_naive(x, w, bias, a, b, scale)
            if eligible("hybrid", m, k, n, r, dtype, False):
                impls["hybrid"] = lambda: _hybrid(x, w, bias, a, b, scale)
            if eligible("concat", m, k, n, r, dtype, True):
                impls["concat"] = lambda: _concat(x, w_aug, b_aug, b, scale, n)
            if eligible("fused", m, k, n, r, dtype, False):
                impls["fused"] = lambda: fused_gemm(x, w, bias, a, b, scale)

            reference = _torch_naive(x, w, bias, a, b, scale).float()
            errors = {name: rel_err(fn(), reference) for name, fn in impls.items() if name != "base"}
            timings = time_all(impls, rounds=args.rounds)
            base_ms = timings["base"]
            rows.append({"shape": label, "m": m, "k": k, "n": n, "variant": "base",
                         "energy": tag, "rank": r, "total_us": base_ms * 1000})

            for name in ORDER:
                if name not in timings:
                    continue
                ms = timings[name]
                row = {
                    "shape": label, "m": m, "k": k, "n": n,
                    "energy": tag, "rank": r, "padded_rank": pad_rank(r),
                    "variant": name, "base_us": base_ms * 1000, "total_us": ms * 1000,
                    "lora_us": (ms - base_ms) * 1000,
                    "overhead_pct": 100.0 * (ms - base_ms) / base_ms,
                    "rel_err": errors[name],
                }
                rows.append(row)
                print(f"{name:8s} {r:5d} {pad_rank(r):4d} {ms*1000:10.1f} "
                      f"{(ms-base_ms)*1000:9.1f} {row['overhead_pct']:7.1f}% {errors[name]:9.2e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {args.output}")

    print("\n=== sidecar overhead vs un-adapted cuBLAS base ===")
    for label, *_ in shapes:
        print(f"\n{label}")
        for name in ORDER:
            cells = []
            for tag, r in RANKS:
                hit = [x for x in rows if x["shape"] == label and x.get("variant") == name
                       and x.get("energy") == tag]
                cells.append(f"{tag}:{hit[0]['overhead_pct']:6.1f}%" if hit else f"{tag}:{'n/a':>7s}")
            print(f"  {name:8s} " + "  ".join(cells))
        best = []
        for tag, r in RANKS:
            hits = [x for x in rows if x["shape"] == label and x.get("energy") == tag
                    and x.get("variant") in ORDER]
            if hits:
                win = min(hits, key=lambda h: h["total_us"])
                best.append(f"{tag}:{win['variant']}={win['overhead_pct']:.0f}%")
        print(f"  {'BEST':8s} " + "  ".join(best))


if __name__ == "__main__":
    main()
