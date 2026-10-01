#!/usr/bin/env python3
"""Correctness tests for the rank-proportional LoRA linear paths.

Every fast path must reproduce the naive ``base + scale * B(A(x))`` formulation
to within the tolerance of the dtype it runs in.  Run under a GPU allocation:

    python -m loraforge_kernels.test_fused_linear
"""

from __future__ import annotations

import sys

import torch
from torch import nn

from loraforge_kernels.fused_linear import (
    FusedLoRALinear,
    _concat,
    _hybrid,
    _kconcat,
    _torch_naive,
    augment_bias,
    augment_weight,
    build_ext,
    expand_add,
    fused_gemm,
    kconcat_weight,
    packed_rank,
    pad_rank,
)
from loraforge_kernels.runtime_lora import attach_fused_lora

TOL = {torch.float16: 4e-3, torch.bfloat16: 3e-2, torch.float32: 1e-4}

FAILURES: list[str] = []


def check(name, got, want, dtype):
    denom = want.abs().max().clamp_min(1e-6)
    err = float((got.float() - want.float()).abs().max() / denom)
    ok = err <= TOL[dtype]
    print(f"  {'PASS' if ok else 'FAIL'} {name:44s} rel_err={err:.3e}")
    if not ok:
        FAILURES.append(f"{name} rel_err={err:.3e} > {TOL[dtype]}")
    return ok


def test_variants_match():
    """Each variant reproduces the naive formulation."""
    print("\n[variants match naive]")
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for m, k, n in ((512, 320, 320), (1024, 1536, 8960), (256, 640, 1280)):
            for r in (1, 7, 16, 17, 40, 64, 130):
                if r > min(k, n):
                    continue
                torch.manual_seed(r + m)
                x = torch.randn(m, k, device="cuda", dtype=dtype) / k**0.5
                w = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
                bias = torch.randn(n, device="cuda", dtype=dtype)
                a = torch.randn(r, k, device="cuda", dtype=dtype) / k**0.5
                b = torch.randn(n, r, device="cuda", dtype=dtype) / r**0.5
                scale = 0.75
                want = _torch_naive(x, w, bias, a, b, scale)
                tag = f"{str(dtype).split('.')[-1]} m{m} k{k} n{n} r{r}"

                check(f"hybrid {tag}", _hybrid(x, w, bias, a, b, scale), want, dtype)
                w_aug = augment_weight(w, a)
                b_aug = augment_bias(bias, r)
                assert w_aug.shape[0] == n + packed_rank(r), w_aug.shape
                check(f"concat {tag}", _concat(x, w_aug, b_aug, b, scale, n), want, dtype)
                if dtype is not torch.float32:
                    w_kc = kconcat_weight(w, b)
                    assert w_kc.shape == (n, k + packed_rank(r)), w_kc.shape
                    check(f"kconcat{tag}", _kconcat(x, w_kc, bias, a, scale, packed_rank(r)),
                          want, dtype)
                if dtype is not torch.float32 and r <= 64:
                    check(f"fused  {tag}", fused_gemm(x, w, bias, a, b, scale), want, dtype)


def test_fused_gemm_without_sidecar():
    """With the sidecar off the fused kernel is a plain linear."""
    print("\n[fused kernel, sidecar disabled]")
    for dtype in (torch.float16, torch.bfloat16):
        m, k, n = 1024, 640, 1280
        x = torch.randn(m, k, device="cuda", dtype=dtype) / k**0.5
        w = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
        bias = torch.randn(n, device="cuda", dtype=dtype)
        check(f"no-lora {str(dtype).split('.')[-1]}",
              fused_gemm(x, w, bias, None, None, 1.0),
              torch.nn.functional.linear(x, w, bias), dtype)


def test_build_ext():
    """The copy-and-shrink pass must reproduce [x | scale * x @ a.T] exactly."""
    print("\n[build_ext]")
    for dtype in (torch.float16, torch.bfloat16):
        for k, r in ((640, 24), (1536, 256), (320, 7)):
            m = 512
            x = torch.randn(m, k, device="cuda", dtype=dtype) / k**0.5
            a = torch.randn(r, k, device="cuda", dtype=dtype) / k**0.5
            pad = packed_rank(r)
            ext = build_ext(x, a, 0.5, pad)
            tag = f"{str(dtype).split('.')[-1]} k{k} r{r}"
            check(f"copy half {tag}", ext[:, :k], x, dtype)
            check(f"shrink half {tag}", ext[:, k : k + r], 0.5 * (x @ a.t()), dtype)
            if ext.shape[1] > k + r:  # packed_rank(r) == r leaves no padding
                check(f"zero pad {tag}", ext[:, k + r :],
                      torch.zeros_like(ext[:, k + r :]), dtype)


def test_expand_add_inplace():
    """In-place and out-of-place epilogues agree."""
    print("\n[expand_add]")
    dtype = torch.float16
    m, n, r = 512, 1280, 24
    y = torch.randn(m, n, device="cuda", dtype=dtype)
    z = torch.randn(m, r, device="cuda", dtype=dtype)
    b = torch.randn(n, r, device="cuda", dtype=dtype) / r**0.5
    want = y + 0.5 * (z @ b.t())
    y0 = y.clone()
    check("expand_add", expand_add(y, z, b, 0.5), want, dtype)
    # It must not have written through to its input.
    check("input untouched", y, y0, dtype)


def test_module_and_swap():
    """FusedLoRALinear and the nn.Linear swap reproduce the sidecar."""
    print("\n[module + swap]")
    dtype = torch.float16
    k, n, r, m = 640, 1280, 24, 2048
    base = nn.Linear(k, n, bias=True).cuda().to(dtype)
    a = (torch.randn(r, k, device="cuda", dtype=dtype) / k**0.5)
    b = (torch.randn(n, r, device="cuda", dtype=dtype) / r**0.5)
    x = torch.randn(4, m // 4, k, device="cuda", dtype=dtype) / k**0.5
    alpha = 0.8
    want = base(x) + alpha * (x @ a.t()) @ b.t()

    module = FusedLoRALinear(base, a, b, scale=alpha)
    check("FusedLoRALinear (3-d input)", module(x), want, dtype)
    assert module.rank == r, module.rank
    assert module.weight.shape == (n, k), module.weight.shape
    check("packed weight view", module.weight, base.weight, dtype)
    check("packed lora_a view", module.lora_a, a, dtype)

    holder = nn.Module()
    holder.inner = nn.Module()
    holder.inner.proj = nn.Linear(k, n, bias=False).cuda().to(dtype)
    w0 = holder.inner.proj.weight.detach().clone()
    attach_fused_lora(holder, "inner.proj", a, b, alpha)
    assert isinstance(holder.inner.proj, FusedLoRALinear)
    check("attach_fused_lora", holder.inner.proj(x), (x @ w0.t()) + alpha * (x @ a.t()) @ b.t(), dtype)

    # Stacking a second adapter must equal the sum of both sidecars.
    r2 = 40
    a2 = torch.randn(r2, k, device="cuda", dtype=dtype) / k**0.5
    b2 = torch.randn(n, r2, device="cuda", dtype=dtype) / r2**0.5
    alpha2 = 0.35
    attach_fused_lora(holder, "inner.proj", a2, b2, alpha2)
    stacked = holder.inner.proj
    assert stacked.rank == r + r2, stacked.rank
    want_stacked = (x @ w0.t()) + alpha * (x @ a.t()) @ b.t() + alpha2 * (x @ a2.t()) @ b2.t()
    check("stacked adapters", stacked(x), want_stacked, dtype)

    # And swapping back to a single low-rank adapter must repack cleanly.
    stacked.set_factors(a, b, scale=alpha)
    assert stacked.rank == r, stacked.rank
    check("set_factors repack", stacked(x), (x @ w0.t()) + alpha * (x @ a.t()) @ b.t(), dtype)


def test_rank_padding():
    print("\n[rank padding]")
    cases = {1: 16, 8: 16, 16: 16, 17: 32, 24: 32, 40: 64, 64: 64, 65: 128}
    for r, want in cases.items():
        got = pad_rank(r)
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'} pad_rank({r}) = {got} (want {want})")
        if not ok:
            FAILURES.append(f"pad_rank({r})={got} want {want}")
    for r, want in {1: 16, 16: 16, 17: 32, 40: 48, 64: 64}.items():
        got = packed_rank(r)
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'} packed_rank({r}) = {got} (want {want})")
        if not ok:
            FAILURES.append(f"packed_rank({r})={got} want {want}")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    test_rank_padding()
    test_build_ext()
    test_expand_add_inplace()
    test_fused_gemm_without_sidecar()
    test_variants_match()
    test_module_and_swap()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
