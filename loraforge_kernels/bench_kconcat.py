#!/usr/bin/env python3
"""Does K-concat beat the N-concat epilogue on the real layer shapes?

  concat  : one wide GEMM [W ; A] then a Triton epilogue over [M, N]  -> +2*M*N
  kconcat : one Triton pass building [x | scale*z] then a wide-K GEMM -> +2*M*K

So the two should trade places around K == N, and kconcat should win big on an
FFN up projection (K << N) and lose on an FFN down projection (K >> N).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from loraforge_kernels.fused_linear import (
    _concat, _hybrid, _kconcat, _torch_naive,
    augment_bias, augment_weight, kconcat_weight, packed_rank,
)

SHAPES = [
    ("wan/attn-1536", 32760, 1536, 1536),
    ("wan/ffn-up-8960", 32760, 1536, 8960),
    ("wan/ffn-down-1536", 32760, 8960, 1536),
    ("llm/prefill-4096", 4096, 4096, 4096),
]
RANKS = [("e100", 256), ("e90", 144), ("e50", 56), ("e70", 16)]
ORDER = ("torch", "hybrid", "concat", "kconcat")


def time_all(fns, warmup=10, rounds=25):
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    best = {k: float("inf") for k in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            b, e = torch.cuda.Event(True), torch.cuda.Event(True)
            b.record(); fn(); e.record(); e.synchronize()
            best[name] = min(best[name], b.elapsed_time(e))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/kconcat.json"))
    args = parser.parse_args()
    dtype, dev = torch.bfloat16, "cuda"
    torch.manual_seed(0)
    rows = []
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for label, m, k, n in SHAPES:
        x = torch.randn(m, k, device=dev, dtype=dtype) / k**0.5
        w = torch.randn(n, k, device=dev, dtype=dtype) / k**0.5
        bias = torch.randn(n, device=dev, dtype=dtype)
        print(f"\n=== {label}  M={m} K={k} N={n}")
        print(f"{'variant':8s} {'rank':>5s} {'total us':>10s} {'lora us':>9s} {'vs base':>8s} {'rel err':>9s}")
        for tag, r in RANKS:
            a = torch.randn(r, k, device=dev, dtype=dtype) / k**0.5
            b = torch.randn(n, r, device=dev, dtype=dtype) / r**0.5
            scale = 1.0
            w_aug, b_aug = augment_weight(w, a), augment_bias(bias, r)
            w_kc = kconcat_weight(w, b)
            rpad = packed_rank(r)

            impls = {
                "base": lambda: torch.nn.functional.linear(x, w, bias),
                "torch": lambda: _torch_naive(x, w, bias, a, b, scale),
                "hybrid": lambda: _hybrid(x, w, bias, a, b, scale),
                "concat": lambda: _concat(x, w_aug, b_aug, b, scale, n),
                "kconcat": lambda: _kconcat(x, w_kc, bias, a, scale, rpad),
            }
            reference = _torch_naive(x, w, bias, a, b, scale).float()
            denom = reference.abs().max().clamp_min(1e-6)
            errs = {nm: float((fn().float() - reference).abs().max() / denom)
                    for nm, fn in impls.items() if nm != "base"}
            t = time_all(impls)
            base = t["base"]
            for nm in ORDER:
                ms = t[nm]
                print(f"{nm:8s} {r:5d} {ms*1000:10.1f} {(ms-base)*1000:9.1f} "
                      f"{100*(ms-base)/base:7.1f}% {errs[nm]:9.2e}")
                rows.append({"shape": label, "m": m, "k": k, "n": n, "rank": r,
                             "energy": tag, "variant": nm, "base_us": base*1000,
                             "total_us": ms*1000, "overhead_pct": 100*(ms-base)/base,
                             "rel_err": errs[nm]})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")
    print("\n=== sidecar overhead vs un-adapted base ===")
    for label, *_ in SHAPES:
        print(f"\n{label}")
        for nm in ORDER:
            cells = [f"{tag}:{next((x['overhead_pct'] for x in rows if x['shape']==label and x['variant']==nm and x['energy']==tag), float('nan')):6.1f}%"
                     for tag, _ in RANKS]
            print(f"  {nm:8s} " + "  ".join(cells))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
