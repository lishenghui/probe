"""Rank-aware Split-K and Persistent Low-Rank GEMM kernels for small-batch LoRA decode."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Hardware-aligned rank bucket choices for GPU deployment
DEPLOY_RANK_BUCKETS = (8, 16, 24, 32, 48, 64)


def align_to_deploy_rank(r: int) -> int:
    """Map arbitrary intrinsic rank to the closest hardware-friendly deployable rank."""
    for bucket in DEPLOY_RANK_BUCKETS:
        if r <= bucket:
            return bucket
    return triton.next_power_of_2(r)


@triton.jit
def _splitk_shrink_kernel(
    x_ptr, a_ptr, z_partial_ptr,
    M: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_ar: tl.constexpr, stride_ak: tl.constexpr,
    stride_zm: tl.constexpr, stride_zk: tl.constexpr, stride_zr: tl.constexpr,
    SPLIT_K: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """Split-K shrink: Grid = (M, SPLIT_K). Distributes huge K dimension across thread blocks."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_chunk_size = (K + SPLIT_K - 1) // SPLIT_K
    k_start = pid_k * k_chunk_size
    k_end = tl.minimum(k_start + k_chunk_size, K)

    rk = tl.arange(0, BR)
    acc = tl.zeros((BR,), dtype=tl.float32)

    for k0 in range(k_start, k_end, BK):
        k_offs = k0 + tl.arange(0, BK)
        k_mask = k_offs < k_end
        
        # Load x: shape (BK,)
        x_val = tl.load(x_ptr + pid_m * stride_xm + k_offs * stride_xk, mask=k_mask, other=0.0)
        
        # Load A: shape (BR, BK)
        a_val = tl.load(
            a_ptr + rk[:, None] * stride_ar + k_offs[None, :] * stride_ak,
            mask=(rk[:, None] < R) & k_mask[None, :],
            other=0.0,
        )
        # acc += A @ x
        acc += tl.sum(a_val * x_val[None, :], axis=1)

    # Write partial reduction
    tl.store(
        z_partial_ptr + pid_m * stride_zm + pid_k * stride_zk + rk * stride_zr,
        acc,
        mask=rk < R,
    )


@triton.jit
def _persistent_expand_add_kernel(
    z_ptr, b_ptr, y_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
    stride_zm: tl.constexpr, stride_zr: tl.constexpr,
    stride_bn: tl.constexpr, stride_br: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    SCALE: tl.constexpr, ADD_Y: tl.constexpr,
    BN: tl.constexpr, BR: tl.constexpr,
):
    """Persistent expand on-chip: Grid = (M, (N + BN - 1) // BN)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rk = tl.arange(0, BR)
    n_offs = pid_n * BN + tl.arange(0, BN)
    n_mask = n_offs < N

    # Load z vector into registers (small: BR elements)
    z_val = tl.load(z_ptr + pid_m * stride_zm + rk * stride_zr, mask=rk < R, other=0.0)

    # Load B: shape (BN, BR)
    b_val = tl.load(
        b_ptr + n_offs[:, None] * stride_bn + rk[None, :] * stride_br,
        mask=n_mask[:, None] & (rk[None, :] < R),
        other=0.0,
    )

    out_val = tl.sum(b_val * z_val[None, :], axis=1) * SCALE

    if ADD_Y:
        y_val = tl.load(y_ptr + pid_m * stride_ym + n_offs * stride_yn, mask=n_mask, other=0.0)
        out_val += y_val

    tl.store(out_ptr + pid_m * stride_om + n_offs * stride_on, out_val, mask=n_mask)


def splitk_lora_shrink(x: torch.Tensor, a: torch.Tensor, split_k: int = 8) -> torch.Tensor:
    """Split-K shrink: x [M, K] @ a [R, K].T -> z [M, R]."""
    m, k = x.shape
    r, ka = a.shape
    assert ka == k and x.is_cuda and a.is_cuda
    
    br = align_to_deploy_rank(r)
    br = triton.next_power_of_2(br)
    
    z_partial = torch.empty((m, split_k, r), device=x.device, dtype=torch.float32)
    grid = (m, split_k)
    
    _splitk_shrink_kernel[grid](
        x, a, z_partial,
        m, k, r,
        x.stride(0), x.stride(1),
        a.stride(0), a.stride(1),
        z_partial.stride(0), z_partial.stride(1), z_partial.stride(2),
        SPLIT_K=split_k, BK=128, BR=br,
        num_warps=4,
    )
    # Sum over split_k dimension
    return z_partial.sum(dim=1).to(x.dtype)


def persistent_expand_add(z: torch.Tensor, b: torch.Tensor, y: torch.Tensor | None = None, scale: float = 1.0) -> torch.Tensor:
    """Persistent expand & add: z [M, R] @ b [N, R].T + y [M, N] -> out [M, N]."""
    m, r = z.shape
    n, rb = b.shape
    assert rb == r and z.is_cuda and b.is_cuda
    
    br = triton.next_power_of_2(align_to_deploy_rank(r))
    bn = 128
    grid = (m, (n + bn - 1) // bn)
    
    out = torch.empty((m, n), device=z.device, dtype=z.dtype)
    y_arg = y if y is not None else out
    
    _persistent_expand_add_kernel[grid](
        z, b, y_arg, out,
        m, n, r,
        z.stride(0), z.stride(1),
        b.stride(0), b.stride(1),
        y_arg.stride(0), y_arg.stride(1),
        out.stride(0), out.stride(1),
        SCALE=scale, ADD_Y=y is not None,
        BN=bn, BR=br,
        num_warps=4,
    )
    return out


def splitk_persistent_fused_lora(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor | None = None, scale: float = 1.0, split_k: int = 8) -> torch.Tensor:
    """Full 2-phase Level 2 fused LoRA with Split-K shrink and persistent expand."""
    z = splitk_lora_shrink(x, a, split_k=split_k)
    return persistent_expand_add(z, b, y=y, scale=scale)
