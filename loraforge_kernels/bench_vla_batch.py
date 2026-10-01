#!/usr/bin/env python3
"""Where does batch size put the VLA sidecar, and when does rank start to matter?

At M=1 the layer is weight-streaming bound and the sidecar is pure launch
overhead -- CUDA graphs remove 93% of it and the rank is irrelevant.  As the
batch grows the base GEMM becomes compute bound and the sidecar's own flops,
which *are* proportional to the rank, should start to show.

The flop ratio says where the ceiling is: sidecar/base = r*(K+N)/(K*N).  For a
rank-64 adapter on a 4096-wide model that is only ~3%, so this also answers
whether the adapter's rank is even large enough for compression to matter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from loraforge_kernels.fused_linear import _concat, _hybrid, _torch_naive, augment_weight

DIM, FFN = 4096, 11008
PROJS = [("q", DIM, DIM), ("k", DIM, DIM), ("v", DIM, DIM), ("o", DIM, DIM),
         ("gate", DIM, FFN), ("up", DIM, FFN), ("down", FFN, DIM)]
BATCHES = [1, 8, 32, 128, 512, 2048]
# OpenVLA ships rank 64 (e95 keeps 19).  256/56 stands in for an adapter whose
# rank is actually large relative to the model width.
RANK_PAIRS = [("openvla r64", 64, 19), ("high-rank r256", 256, 56)]


def time_fn(fn, warmup=8, rounds=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record(); fn(); e.record(); e.synchronize()
        best = min(best, b.elapsed_time(e))
    return best


def capture(fn, warmup=12):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=torch.cuda.graphs.graph_pool_handle()):
        fn()
    return g


class Layer:
    def __init__(self, m, rank, dtype, dev):
        self.xs, self.w, self.a, self.b, self.w_aug = [], [], [], [], []
        for _, k, n in PROJS:
            self.xs.append(torch.randn(m, k, device=dev, dtype=dtype) / k**0.5)
            self.w.append(torch.randn(n, k, device=dev, dtype=dtype) / k**0.5)
            a = torch.randn(rank, k, device=dev, dtype=dtype) / k**0.5
            b = torch.randn(n, rank, device=dev, dtype=dtype) / rank**0.5
            self.a.append(a); self.b.append(b)
            self.w_aug.append(augment_weight(self.w[-1], a))
        self.scale = 16.0 / rank

    def base(self):
        return [F.linear(x, w) for x, w in zip(self.xs, self.w)]

    def best(self):
        """concat where it applies, hybrid otherwise -- what the selector picks."""
        return [_concat(x, wa, None, b, self.scale, w.shape[0])
                for x, wa, b, w in zip(self.xs, self.w_aug, self.b, self.w)]

    def naive(self):
        return [_torch_naive(x, w, None, a, b, self.scale)
                for x, w, a, b in zip(self.xs, self.w, self.a, self.b)]


def flop_ceiling(rank):
    """sidecar/base flop ratio for this layer, the compute-bound ceiling."""
    num = den = 0
    for _, k, n in PROJS:
        num += rank * (k + n)
        den += k * n
    return 100.0 * num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/vla_batch.json"))
    args = parser.parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}   llama2-7b layer, dim={DIM} ffn={FFN}")
    rows = []

    for label, hi, lo in RANK_PAIRS:
        print(f"\n=== {label}: rank {hi} vs {lo}   "
              f"(flop ceiling {flop_ceiling(hi):.1f}% -> {flop_ceiling(lo):.1f}%)")
        print(f"{'batch':>6s} {'base us':>9s} {'graphed':>9s} | "
              f"{'naive r'+str(hi):>11s} {'sidecar':>8s} | "
              f"{'best r'+str(hi):>10s} {'sidecar':>8s} | {'best r'+str(lo):>10s} {'sidecar':>8s} | "
              f"{'gain':>6s}")
        for m in BATCHES:
            hi_layer = Layer(m, hi, dtype, dev)
            lo_layer = Layer(m, lo, dtype, dev)
            g_base = capture(hi_layer.base)
            g_naive = capture(hi_layer.naive)
            g_hi = capture(hi_layer.best)
            g_lo = capture(lo_layer.best)
            eager_base = time_fn(hi_layer.base)
            t_base = time_fn(g_base.replay)
            t_naive = time_fn(g_naive.replay)
            t_hi = time_fn(g_hi.replay)
            t_lo = time_fn(g_lo.replay)
            s_naive = 100 * (t_naive - t_base) / t_base
            s_hi = 100 * (t_hi - t_base) / t_base
            s_lo = 100 * (t_lo - t_base) / t_base
            print(f"{m:6d} {eager_base*1000:9.1f} {t_base*1000:9.1f} | "
                  f"{t_naive*1000:11.1f} {s_naive:7.1f}% | "
                  f"{t_hi*1000:10.1f} {s_hi:7.1f}% | {t_lo*1000:10.1f} {s_lo:7.1f}% | "
                  f"{100*(t_hi/t_lo-1):+5.1f}%")
            rows.append({"pair": label, "batch": m, "rank_hi": hi, "rank_lo": lo,
                         "eager_base_us": eager_base * 1000, "base_us": t_base * 1000,
                         "naive_us": t_naive * 1000, "hi_us": t_hi * 1000, "lo_us": t_lo * 1000,
                         "sidecar_naive_pct": s_naive, "sidecar_hi_pct": s_hi,
                         "sidecar_lo_pct": s_lo, "gain_pct": 100 * (t_hi / t_lo - 1)})
            del hi_layer, lo_layer, g_base, g_naive, g_hi, g_lo
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
