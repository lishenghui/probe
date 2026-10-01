#!/usr/bin/env python3
"""Why a smaller LoRA rank can cost more: cuBLAS efficiency vs the rank.

The sidecar's shrink is x[M,K] @ A[r,K].T -- a GEMM whose output dimension is
the rank.  Compressing an adapter makes that dimension small and, after
per-module energy truncation, arbitrary.  This measures what cuBLAS actually
delivers as the rank shrinks, on the Wan2.1 attention shape.
"""
import statistics
import torch

def t_us(fn, warmup=20, reps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(reps):
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record(); fn(); e.record(); e.synchronize()
        s.append(b.elapsed_time(e) * 1000)
    return min(s)

dev, dt = "cuda", torch.bfloat16
M, K, N = 32760, 1536, 1536
x = torch.randn(M, K, device=dev, dtype=dt) / K**0.5
w = torch.randn(N, K, device=dev, dtype=dt) / K**0.5
base = t_us(lambda: torch.nn.functional.linear(x, w))
print(f"{torch.cuda.get_device_name(0)}  M={M} K={K} N={N}")
print(f"base GEMM (N={N}): {base:8.1f} us  ->  "
      f"{2*M*K*N/base*1e-6:7.1f} TFLOP/s\n")

# The bytes the shrink must move are the same for every rank: it streams x.
x_bytes = M * K * 2
print(f"{'rank':>5s} {'shrink us':>10s} {'TFLOP/s':>9s} {'GB/s':>8s} "
      f"{'expand us':>10s} {'total us':>9s}   note")
for r in (256, 241, 226, 170, 142, 113, 51, 33, 17, 11, 8, 3):
    a = torch.randn(r, K, device=dev, dtype=dt) / K**0.5
    b = torch.randn(N, r, device=dev, dtype=dt) / r**0.5
    shrink = t_us(lambda: torch.mm(x, a.t()))
    z = torch.mm(x, a.t())
    expand = t_us(lambda: torch.mm(z, b.t()))
    note = "aligned16" if r % 16 == 0 else ("aligned8" if r % 8 == 0 else "unaligned")
    print(f"{r:5d} {shrink:10.1f} {2*M*K*r/shrink*1e-6:9.1f} "
          f"{x_bytes/shrink*1e-3:8.0f} {expand:10.1f} {shrink+expand:9.1f}   {note}")
