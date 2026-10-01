#!/usr/bin/env python3
"""Is a fast path actually less accurate than the naive one, or is TOL tight?

Comparing two fp16 paths against each other cannot answer that -- it is how a
real miscompile in the old Triton copy-and-shrink kernel first looked like a
tolerance question.  Compare each against float64 instead.
"""
import torch



def accuracy_vs_fp64():
    """Is kconcat actually less accurate than the naive path, or is TOL tight?

    Both are fp16; compare each against a float64 evaluation of the same maths
    rather than against each other.
    """
    from loraforge_kernels.fused_linear import _kconcat, _torch_naive, kconcat_weight, packed_rank
    print("\n=== accuracy vs float64 reference ===")
    print(f"{'shape':22s} {'rank':>5s} {'naive':>10s} {'kconcat':>10s}")
    for m, k, n in ((1024, 1536, 8960), (256, 640, 1280)):
        for r in (40, 64, 130):
            torch.manual_seed(r + m)
            dtype = torch.float16
            x = torch.randn(m, k, device="cuda", dtype=dtype) / k**0.5
            w = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
            bias = torch.randn(n, device="cuda", dtype=dtype)
            a = torch.randn(r, k, device="cuda", dtype=dtype) / k**0.5
            b = torch.randn(n, r, device="cuda", dtype=dtype) / r**0.5
            scale = 0.75
            exact = (x.double() @ w.double().t() + bias.double()
                     + scale * (x.double() @ a.double().t()) @ b.double().t())
            denom = exact.abs().max().clamp_min(1e-12)
            naive = float((_torch_naive(x, w, bias, a, b, scale).double() - exact).abs().max() / denom)
            kc = float((_kconcat(x, kconcat_weight(w, b), bias, a, scale, packed_rank(r)).double()
                        - exact).abs().max() / denom)
            print(f"{f'm{m} k{k} n{n}':22s} {r:5d} {naive:10.2e} {kc:10.2e}")


if __name__ == "__main__":
    accuracy_vs_fp64()
