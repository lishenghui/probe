"""Triton prototypes for fixed-overhead-dominated LoRA decode kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_lora_row_kernel(
    x, a, b, y, out,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_ar: tl.constexpr, stride_ak: tl.constexpr,
    stride_bn: tl.constexpr, stride_br: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    SCALE: tl.constexpr, ADD_Y: tl.constexpr,
    BK: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
):
    """One persistent program per token row; XA stays in registers."""
    row = tl.program_id(0)
    rk = tl.arange(0, BR)
    z = tl.zeros((BR,), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        xv = tl.load(x + row * stride_xm + kk * stride_xk, mask=kk < K, other=0.0)
        av = tl.load(
            a + rk[:, None] * stride_ar + kk[None, :] * stride_ak,
            mask=(rk[:, None] < R) & (kk[None, :] < K), other=0.0,
        )
        z += tl.sum(av * xv[None, :], axis=1)
    for n0 in range(0, N, BN):
        nn = n0 + tl.arange(0, BN)
        bv = tl.load(
            b + nn[:, None] * stride_bn + rk[None, :] * stride_br,
            mask=(nn[:, None] < N) & (rk[None, :] < R), other=0.0,
        )
        update = tl.sum(bv * z[None, :], axis=1) * SCALE
        if ADD_Y:
            update += tl.load(y + row * stride_ym + nn * stride_yn, mask=nn < N)
        tl.store(out + row * stride_om + nn * stride_on, update, mask=nn < N)


@triton.jit
def _fused_lora_output_tile_kernel(
    x, a, b, y, out,
    K: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_ar: tl.constexpr, stride_ak: tl.constexpr,
    stride_bn: tl.constexpr, stride_br: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    SCALE: tl.constexpr, ADD_Y: tl.constexpr,
    BK: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
):
    """One program per (token, output tile) to expose decode parallelism.

    Each output tile recomputes x@A. That is intentionally redundant: for
    small compressed ranks the shrink is cheap, while M*ceil(N/BN) programs
    can occupy a large GPU where the one-program-per-row kernel cannot.
    """
    row = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    rk = tl.arange(0, BR)
    z = tl.zeros((BR,), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        xv = tl.load(x + row * stride_xm + kk * stride_xk, mask=kk < K, other=0.0)
        av = tl.load(
            a + rk[:, None] * stride_ar + kk[None, :] * stride_ak,
            mask=(rk[:, None] < R) & (kk[None, :] < K), other=0.0,
        )
        z += tl.sum(av * xv[None, :], axis=1)
    nn = n0 + tl.arange(0, BN)
    bv = tl.load(
        b + nn[:, None] * stride_bn + rk[None, :] * stride_br,
        mask=(nn[:, None] < N) & (rk[None, :] < R), other=0.0,
    )
    update = tl.sum(bv * z[None, :], axis=1) * SCALE
    if ADD_Y:
        update += tl.load(y + row * stride_ym + nn * stride_yn, mask=nn < N)
    tl.store(out + row * stride_om + nn * stride_on, update, mask=nn < N)


@triton.jit(do_not_specialize=["actual_r", "stride_bn"])
def _classed_lora_row_kernel(
    x, a, b, y, out, actual_r,
    stride_xm, stride_xk, stride_ar, stride_ak, stride_bn, stride_br,
    stride_ym, stride_yn, stride_om, stride_on,
    K: tl.constexpr, N: tl.constexpr,
    SCALE: tl.constexpr, ADD_Y: tl.constexpr,
    BK: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
):
    """One executable class for every physical rank covered by ``BR``."""
    row = tl.program_id(0)
    rk = tl.arange(0, BR)
    rank_mask = rk < actual_r
    z = tl.zeros((BR,), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        k_mask = kk < K
        xv = tl.load(x + row * stride_xm + kk * stride_xk, mask=k_mask, other=0.0)
        av = tl.load(
            a + rk[:, None] * stride_ar + kk[None, :] * stride_ak,
            mask=rank_mask[:, None] & k_mask[None, :], other=0.0,
        )
        z += tl.sum(av * xv[None, :], axis=1)
    for n0 in range(0, N, BN):
        nn = n0 + tl.arange(0, BN)
        n_mask = nn < N
        bv = tl.load(
            b + nn[:, None] * stride_bn + rk[None, :] * stride_br,
            mask=n_mask[:, None] & rank_mask[None, :], other=0.0,
        )
        update = tl.sum(bv * z[None, :], axis=1) * SCALE
        if ADD_Y:
            update += tl.load(y + row * stride_ym + nn * stride_yn, mask=n_mask)
        tl.store(out + row * stride_om + nn * stride_on, update, mask=n_mask)


@triton.jit(do_not_specialize=["actual_r", "stride_bn"])
def _classed_lora_output_tile_kernel(
    x, a, b, y, out, actual_r,
    stride_xm, stride_xk, stride_ar, stride_ak, stride_bn, stride_br,
    stride_ym, stride_yn, stride_om, stride_on,
    K: tl.constexpr, N: tl.constexpr,
    SCALE: tl.constexpr, ADD_Y: tl.constexpr,
    BK: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
):
    """Output-parallel counterpart sharing one executable per ``BR`` class."""
    row = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    rk = tl.arange(0, BR)
    rank_mask = rk < actual_r
    z = tl.zeros((BR,), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        k_mask = kk < K
        xv = tl.load(x + row * stride_xm + kk * stride_xk, mask=k_mask, other=0.0)
        av = tl.load(
            a + rk[:, None] * stride_ar + kk[None, :] * stride_ak,
            mask=rank_mask[:, None] & k_mask[None, :], other=0.0,
        )
        z += tl.sum(av * xv[None, :], axis=1)
    nn = n0 + tl.arange(0, BN)
    n_mask = nn < N
    bv = tl.load(
        b + nn[:, None] * stride_bn + rk[None, :] * stride_br,
        mask=n_mask[:, None] & rank_mask[None, :], other=0.0,
    )
    update = tl.sum(bv * z[None, :], axis=1) * SCALE
    if ADD_Y:
        update += tl.load(y + row * stride_ym + nn * stride_yn, mask=n_mask)
    tl.store(out + row * stride_om + nn * stride_on, update, mask=n_mask)


@triton.jit
def _grouped_lora_row_kernel(
    x, a, b, y, out,
    K: tl.constexpr, N: tl.constexpr, R: tl.constexpr, P: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_ar: tl.constexpr, stride_ak: tl.constexpr,
    stride_bp: tl.constexpr, stride_bn: tl.constexpr, stride_br: tl.constexpr,
    stride_ym: tl.constexpr, stride_yp: tl.constexpr, stride_yn: tl.constexpr,
    stride_om: tl.constexpr, stride_op: tl.constexpr, stride_on: tl.constexpr,
    SCALE: tl.constexpr, ADD_Y: tl.constexpr,
    BK: tl.constexpr, BN: tl.constexpr, BPR: tl.constexpr,
):
    """Fuse P projections by concatenating their A ranks; X is loaded once."""
    row = tl.program_id(0)
    pr = tl.arange(0, BPR)
    z = tl.zeros((BPR,), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        xv = tl.load(x + row * stride_xm + kk * stride_xk, mask=kk < K, other=0.0)
        av = tl.load(
            a + pr[:, None] * stride_ar + kk[None, :] * stride_ak,
            mask=(pr[:, None] < P * R) & (kk[None, :] < K), other=0.0,
        )
        z += tl.sum(av * xv[None, :], axis=1)
    for p in range(P):
        for n0 in range(0, N, BN):
            nn = n0 + tl.arange(0, BN)
            local_r = pr - p * R
            bv = tl.load(
                b + p * stride_bp + nn[:, None] * stride_bn + local_r[None, :] * stride_br,
                mask=(nn[:, None] < N) & (local_r[None, :] >= 0) & (local_r[None, :] < R),
                other=0.0,
            )
            update = tl.sum(bv * z[None, :], axis=1) * SCALE
            if ADD_Y:
                update += tl.load(
                    y + row * stride_ym + p * stride_yp + nn * stride_yn, mask=nn < N
                )
            tl.store(
                out + row * stride_om + p * stride_op + nn * stride_on,
                update, mask=nn < N,
            )


def fused_lora(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y=None, scale=1.0):
    m, k = x.shape
    r, ka = a.shape
    n, rb = b.shape
    assert ka == k and rb == r and r <= 64 and x.is_cuda
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    y_arg = y if y is not None else out
    _fused_lora_row_kernel[(m,)](
        x, a, b, y_arg, out, m, k, n, r,
        x.stride(0), x.stride(1), a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        y_arg.stride(0), y_arg.stride(1), out.stride(0), out.stride(1),
        SCALE=scale, ADD_Y=y is not None, BK=128, BN=128, BR=triton.next_power_of_2(r),
        num_warps=4,
    )
    return out


def tiled_lora(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y=None, scale=1.0):
    """Output-parallel decode kernel; best suited to small M and compressed R."""
    m, k = x.shape
    r, ka = a.shape
    n, rb = b.shape
    assert ka == k and rb == r and r <= 64 and x.is_cuda
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    y_arg = y if y is not None else out
    bn = 128
    _fused_lora_output_tile_kernel[(m, triton.cdiv(n, bn))](
        x, a, b, y_arg, out, k, n, r,
        x.stride(0), x.stride(1), a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        y_arg.stride(0), y_arg.stride(1), out.stride(0), out.stride(1),
        SCALE=scale, ADD_Y=y is not None, BK=128, BN=bn,
        BR=triton.next_power_of_2(r), num_warps=4,
    )
    return out


def _validate_execution_class(r: int, execution_rank: int | None) -> int:
    execution_rank = triton.next_power_of_2(r) if execution_rank is None else execution_rank
    assert r <= execution_rank <= 64
    assert execution_rank & (execution_rank - 1) == 0
    return execution_rank


def classed_lora(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y=None, scale=1.0,
    execution_rank: int | None = None,
):
    """Compact physical storage mapped to a shared row-kernel execution class."""
    m, k = x.shape
    r, ka = a.shape
    n, rb = b.shape
    assert ka == k and rb == r and x.is_cuda
    br = _validate_execution_class(r, execution_rank)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    y_arg = y if y is not None else out
    _classed_lora_row_kernel[(m,)](
        x, a, b, y_arg, out, r,
        x.stride(0), x.stride(1), a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), y_arg.stride(0), y_arg.stride(1),
        out.stride(0), out.stride(1),
        K=k, N=n, SCALE=scale, ADD_Y=y is not None, BK=128, BN=128, BR=br,
        num_warps=4,
    )
    return out


def classed_tiled_lora(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y=None, scale=1.0,
    execution_rank: int | None = None,
):
    """Compact physical storage mapped to a shared output-tiled execution class."""
    m, k = x.shape
    r, ka = a.shape
    n, rb = b.shape
    assert ka == k and rb == r and x.is_cuda
    br = _validate_execution_class(r, execution_rank)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    y_arg = y if y is not None else out
    bn = 128
    _classed_lora_output_tile_kernel[(m, triton.cdiv(n, bn))](
        x, a, b, y_arg, out, r,
        x.stride(0), x.stride(1), a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), y_arg.stride(0), y_arg.stride(1),
        out.stride(0), out.stride(1),
        K=k, N=n, SCALE=scale, ADD_Y=y is not None, BK=128, BN=bn, BR=br,
        num_warps=4,
    )
    return out


def grouped_lora(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y=None, scale=1.0):
    """a: [P*R,K], b: [P,N,R], y/out: [M,P,N]."""
    m, k = x.shape
    p, n, r = b.shape
    assert a.shape == (p * r, k) and p * r <= 128 and x.is_cuda
    out = torch.empty((m, p, n), device=x.device, dtype=x.dtype)
    y_arg = y if y is not None else out
    _grouped_lora_row_kernel[(m,)](
        x, a, b, y_arg, out, k, n, r, p,
        x.stride(0), x.stride(1), a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        y_arg.stride(0), y_arg.stride(1), y_arg.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        SCALE=scale, ADD_Y=y is not None, BK=128, BN=128,
        BPR=triton.next_power_of_2(p * r), num_warps=8,
    )
    return out
