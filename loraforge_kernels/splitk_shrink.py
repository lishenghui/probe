"""Split-K shrink for the skinny GEMM at the head of a LoRA sidecar.

The shrink is ``x[M, K] @ A[r, K].T``.  Its output is ``M x r``, and r is 8-64,
so a standard GEMM has very few output tiles to hand out: at small M the kernel
runs on a handful of SMs no matter how big K is.  Splitting the long K
dimension across thread blocks manufactures parallel work units, at the cost of
a reduction over the partials.

Two ways to pay for that reduction, and which wins is not obvious:

``partial``  each block writes its own slice of an ``[M, SPLIT_K, r]`` buffer and
             a second pass sums it.  Costs one extra kernel launch and
             ``M*SPLIT_K*r`` of traffic -- negligible bytes at small M, but the
             launch is not negligible there.
``atomic``   each block atomically accumulates into one ``[M, r]`` fp32 buffer.
             One launch, no partial buffer, but contention grows with SPLIT_K.

Both are benchmarked against cuBLAS in bench_splitk.py; nothing here assumes a
winner.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def rank_tile(r: int) -> int:
    """Tensor-core tiles need at least 16 along the contracted dimension."""
    return max(16, triton.next_power_of_2(r))


@triton.jit
def _splitk_shrink_partial(
    x_ptr, a_ptr, out_ptr,
    M, K, R,
    stride_xm, stride_xk,
    stride_ar, stride_ak,
    stride_os, stride_om, stride_or,
    SPLIT_K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """Partial sums into out[SPLIT_K, M, R]; a second pass reduces over dim 0."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_r = tl.arange(0, BR)
    mask_m = offs_m < M
    mask_r = offs_r < R

    chunk = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * chunk
    k_end = tl.minimum(k_start + chunk, K)

    offs_k = k_start + tl.arange(0, BK)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    a_ptrs = a_ptr + offs_r[None, :] * stride_ar + offs_k[:, None] * stride_ak

    acc = tl.zeros((BM, BR), dtype=tl.float32)
    for k0 in range(k_start, k_end, BK):
        mask_k = (offs_k < k_end) & (offs_k >= k_start)
        xt = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        at = tl.load(a_ptrs, mask=mask_k[:, None] & mask_r[None, :], other=0.0)
        acc = tl.dot(xt, at, acc)
        offs_k += BK
        x_ptrs += BK * stride_xk
        a_ptrs += BK * stride_ak

    tl.store(
        out_ptr + pid_k * stride_os + offs_m[:, None] * stride_om + offs_r[None, :] * stride_or,
        acc, mask=mask_m[:, None] & mask_r[None, :],
    )


@triton.jit
def _splitk_shrink_atomic(
    x_ptr, a_ptr, out_ptr,
    M, K, R,
    stride_xm, stride_xk,
    stride_ar, stride_ak,
    stride_om, stride_or,
    SPLIT_K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """Accumulate straight into out[M, R] (fp32, pre-zeroed) with atomics."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_r = tl.arange(0, BR)
    mask_m = offs_m < M
    mask_r = offs_r < R

    chunk = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * chunk
    k_end = tl.minimum(k_start + chunk, K)

    offs_k = k_start + tl.arange(0, BK)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    a_ptrs = a_ptr + offs_r[None, :] * stride_ar + offs_k[:, None] * stride_ak

    acc = tl.zeros((BM, BR), dtype=tl.float32)
    for k0 in range(k_start, k_end, BK):
        mask_k = (offs_k < k_end) & (offs_k >= k_start)
        xt = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        at = tl.load(a_ptrs, mask=mask_k[:, None] & mask_r[None, :], other=0.0)
        acc = tl.dot(xt, at, acc)
        offs_k += BK
        x_ptrs += BK * stride_xk
        a_ptrs += BK * stride_ak

    tl.atomic_add(
        out_ptr + offs_m[:, None] * stride_om + offs_r[None, :] * stride_or,
        acc, mask=mask_m[:, None] & mask_r[None, :],
    )


def shrink_partial(x, a, split_k=4, bm=16, bk=128, num_warps=4):
    m, k = x.shape
    r = a.shape[0]
    br = rank_tile(r)
    partial = torch.empty((split_k, m, r), device=x.device, dtype=torch.float32)
    _splitk_shrink_partial[(triton.cdiv(m, bm), split_k)](
        x, a, partial, m, k, r,
        x.stride(0), x.stride(1), a.stride(0), a.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2),
        SPLIT_K=split_k, BM=bm, BK=bk, BR=br, num_warps=num_warps,
    )
    return partial.sum(0).to(x.dtype)


def shrink_atomic(x, a, split_k=4, bm=16, bk=128, num_warps=4):
    m, k = x.shape
    r = a.shape[0]
    br = rank_tile(r)
    out = torch.zeros((m, r), device=x.device, dtype=torch.float32)
    _splitk_shrink_atomic[(triton.cdiv(m, bm), split_k)](
        x, a, out, m, k, r,
        x.stride(0), x.stride(1), a.stride(0), a.stride(1),
        out.stride(0), out.stride(1),
        SPLIT_K=split_k, BM=bm, BK=bk, BR=br, num_warps=num_warps,
    )
    return out.to(x.dtype)
