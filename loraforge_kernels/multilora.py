"""Batched multi-adapter LoRA, where the adapter weights are the traffic.

Single-adapter inference amortises A and B over every row of the batch, which is
why its cost barely tracks the rank.  Multi-tenant serving does not: each token
uses its own adapter, so the LoRA weights are streamed per adapter and the cost
is proportional to the ranks actually present in the batch.

That makes the batch's rank composition a first-class scheduling quantity, and
it is where compression can finally pay -- provided the kernel does not throw
the saving away by padding every adapter to the batch's maximum rank, which is
what a uniform-rank kernel has to do.

``shrink_padded``  one kernel over a [N, r_max, K] stack: simple, one launch,
                   and every adapter pays r_max whatever its own rank is.
``shrink_grouped`` adapters bucketed by rank, one launch per distinct rank, each
                   at its own width.  Pays sum(r_i) of traffic but costs a launch
                   per bucket and a gather of the tokens into it -- measured, that
                   loses to padding at serving batch sizes.
``shrink_ragged``  one launch, no gather: each token looks up its own adapter's
                   rank and offset in a flat [sum(r_i), K] table and reads only
                   that many rows.  This is the one that turns the traffic saving
                   into wall clock.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def rank_tile(r: int) -> int:
    return max(16, triton.next_power_of_2(r))


def auto_split_k(n_tokens: int, target_programs: int = 2048, cap: int = 64) -> int:
    """Split K far enough to keep the machine busy for this many tokens.

    Rank bucketing narrows the tile but shrinks each launch: a bucket holding 32
    of 128 tokens at a fixed split of 8 is only 256 programs, which does not
    fill 132 SMs, and the occupancy lost costs more than the tile width saved.
    Splitting the long K dimension is how a small bucket gets its parallelism
    back -- this is the case the technique is actually for, unlike a dense
    single-adapter shrink where M already supplies plenty of tiles.
    """
    if n_tokens <= 0:
        return 1
    return max(1, min(cap, triton.next_power_of_2(max(1, target_programs // n_tokens))))


@triton.jit
def _bgmv_shrink_kernel(
    x_ptr, a_ptr, idx_ptr, out_ptr,
    T, K, R,
    stride_xt, stride_xk,
    stride_an, stride_ar, stride_ak,
    stride_ot, stride_or,
    SPLIT_K: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """z[t] = x[t] @ A[idx[t]].T, one program per (token, K-slice).

    The output is only T x R, so at serving batch sizes there are very few tiles
    to hand out; SPLIT_K exists to manufacture more of them out of the long K.
    Partial sums are accumulated with atomics into a pre-zeroed fp32 buffer.
    """
    t = tl.program_id(0)
    pid_k = tl.program_id(1)
    adapter = tl.load(idx_ptr + t)

    offs_r = tl.arange(0, BR)
    mask_r = offs_r < R
    chunk = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * chunk
    k_end = tl.minimum(k_start + chunk, K)

    offs_k = k_start + tl.arange(0, BK)
    acc = tl.zeros((BR,), dtype=tl.float32)
    for _ in range(0, tl.cdiv(chunk, BK)):
        mask_k = (offs_k < k_end) & (offs_k >= k_start)
        xv = tl.load(x_ptr + t * stride_xt + offs_k * stride_xk, mask=mask_k, other=0.0)
        av = tl.load(
            a_ptr + adapter * stride_an + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak,
            mask=mask_r[:, None] & mask_k[None, :], other=0.0,
        )
        acc += tl.sum(av * xv[None, :].to(av.dtype), axis=1)
        offs_k += BK

    tl.atomic_add(out_ptr + t * stride_ot + offs_r * stride_or, acc, mask=mask_r)


def shrink_padded(x, a_stack, idx, split_k=None, bk=128):
    """Every adapter padded to the stack's rank; the batch pays r_max per token."""
    t, k = x.shape
    r = a_stack.shape[1]
    split_k = auto_split_k(t) if split_k is None else split_k
    out = torch.zeros((t, r), device=x.device, dtype=torch.float32)
    _bgmv_shrink_kernel[(t, split_k)](
        x, a_stack, idx, out, t, k, r,
        x.stride(0), x.stride(1),
        a_stack.stride(0), a_stack.stride(1), a_stack.stride(2),
        out.stride(0), out.stride(1),
        SPLIT_K=split_k, BK=bk, BR=rank_tile(r), num_warps=4,
    )
    return out


def shrink_grouped(x, groups, idx, split_k=8, bk=128):
    """One launch per distinct rank, each at its own width.

    ``groups`` maps rank -> (a_stack [n_r, rank, K], token positions, local index).
    Only the ranks actually present are paid for.
    """
    outs = {}
    for rank, (a_stack, positions, local_idx) in groups.items():
        xs = x[positions]
        out = torch.zeros((xs.shape[0], rank), device=x.device, dtype=torch.float32)
        _bgmv_shrink_kernel[(xs.shape[0], split_k)](
            xs, a_stack, local_idx, out, xs.shape[0], x.shape[1], rank,
            xs.stride(0), xs.stride(1),
            a_stack.stride(0), a_stack.stride(1), a_stack.stride(2),
            out.stride(0), out.stride(1),
            SPLIT_K=split_k, BK=bk, BR=rank_tile(rank), num_warps=4,
        )
        outs[rank] = (out, positions)
    return outs


@triton.jit
def _ragged_shrink_kernel(
    x_ptr, a_ptr, idx_ptr, off_ptr, rank_ptr, out_ptr,
    T, K, R_MAX,
    stride_xt, stride_xk,
    stride_ar, stride_ak,
    stride_ot, stride_or,
    SPLIT_K: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """z[t, :r_t] = x[t] @ A[off_t : off_t + r_t].T, one launch for the whole batch.

    The adapters live end to end in one [sum(r_i), K] table, so a token reads
    exactly its own rank's worth of rows -- no padding to the batch maximum and
    no gathering tokens into per-rank buckets.  The output is padded to R_MAX,
    which costs nothing: it is T x R_MAX of tiny writes, not the weight traffic.
    """
    t = tl.program_id(0)
    pid_k = tl.program_id(1)
    adapter = tl.load(idx_ptr + t)
    off = tl.load(off_ptr + adapter)
    rank = tl.load(rank_ptr + adapter)

    offs_r = tl.arange(0, BR)
    mask_r = offs_r < rank
    chunk = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * chunk
    k_end = tl.minimum(k_start + chunk, K)

    offs_k = k_start + tl.arange(0, BK)
    acc = tl.zeros((BR,), dtype=tl.float32)
    for _ in range(0, tl.cdiv(chunk, BK)):
        mask_k = (offs_k < k_end) & (offs_k >= k_start)
        xv = tl.load(x_ptr + t * stride_xt + offs_k * stride_xk, mask=mask_k, other=0.0)
        av = tl.load(
            a_ptr + (off + offs_r[:, None]) * stride_ar + offs_k[None, :] * stride_ak,
            mask=mask_r[:, None] & mask_k[None, :], other=0.0,
        )
        acc += tl.sum(av * xv[None, :].to(av.dtype), axis=1)
        offs_k += BK

    tl.atomic_add(out_ptr + t * stride_ot + offs_r * stride_or, acc, mask=mask_r)


def shrink_ragged(x, a_flat, offsets, ranks, idx, r_max, split_k=8, bk=128):
    """Rank-aware shrink: one launch, each token pays only its own rank."""
    t, k = x.shape
    out = torch.zeros((t, r_max), device=x.device, dtype=torch.float32)
    _ragged_shrink_kernel[(t, split_k)](
        x, a_flat, idx, offsets, ranks, out, t, k, r_max,
        x.stride(0), x.stride(1),
        a_flat.stride(0), a_flat.stride(1),
        out.stride(0), out.stride(1),
        SPLIT_K=split_k, BK=bk, BR=rank_tile(r_max), num_warps=4,
    )
    return out


@triton.jit
def _bucket_shrink_kernel(
    x_ptr, a_ptr, pos_ptr, local_ptr, out_ptr,
    NPOS, K, R,
    stride_xt, stride_xk,
    stride_an, stride_ar, stride_ak,
    stride_ot, stride_or,
    SPLIT_K: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """One rank bucket, tiled at exactly that rank, reading tokens by index.

    Masking a wide tile down to a small rank saves the memory traffic but not
    the tile: measured, what tracks the time is BR itself (uniform r64 -> r16 is
    3.2x, while masking a 32-wide tile down to rank 4 bought 4%).  So each
    bucket gets a kernel compiled at its own BR, and the tokens belonging to it
    are addressed through `pos` rather than gathered into a new tensor -- the
    gather is what made the earlier bucketed version lose to padding.
    """
    i = tl.program_id(0)
    pid_k = tl.program_id(1)
    t = tl.load(pos_ptr + i)
    adapter = tl.load(local_ptr + i)

    offs_r = tl.arange(0, BR)
    mask_r = offs_r < R
    chunk = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * chunk
    k_end = tl.minimum(k_start + chunk, K)

    offs_k = k_start + tl.arange(0, BK)
    acc = tl.zeros((BR,), dtype=tl.float32)
    for _ in range(0, tl.cdiv(chunk, BK)):
        mask_k = (offs_k < k_end) & (offs_k >= k_start)
        xv = tl.load(x_ptr + t * stride_xt + offs_k * stride_xk, mask=mask_k, other=0.0)
        av = tl.load(
            a_ptr + adapter * stride_an + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak,
            mask=mask_r[:, None] & mask_k[None, :], other=0.0,
        )
        acc += tl.sum(av * xv[None, :].to(av.dtype), axis=1)
        offs_k += BK

    tl.atomic_add(out_ptr + t * stride_ot + offs_r * stride_or, acc, mask=mask_r)


def shrink_bucketed(x, buckets, r_max, split_k=None, bk=128):
    """Rank-aware shrink: one launch per distinct rank, no gather.

    ``buckets`` maps rank -> (a_stack [n_r, rank, K], token positions, local index).
    Inside a CUDA graph the extra launches are nearly free, so this keeps the
    narrow tiles without paying for them.
    """
    t = x.shape[0]
    out = torch.zeros((t, r_max), device=x.device, dtype=torch.float32)
    for rank, (a_stack, pos, local) in buckets.items():
        n = pos.shape[0]
        if n == 0:
            continue
        sk = auto_split_k(n) if split_k is None else split_k
        _bucket_shrink_kernel[(n, sk)](
            x, a_stack, pos, local, out, n, x.shape[1], rank,
            x.stride(0), x.stride(1),
            a_stack.stride(0), a_stack.stride(1), a_stack.stride(2),
            out.stride(0), out.stride(1),
            SPLIT_K=sk, BK=bk, BR=rank_tile(rank), num_warps=4,
        )
    return out


def shrink_bucketed_parallel(x, buckets, r_max, streams, events, split_k=1, bk=128):
    """Rank buckets on concurrent streams, so the batch pays max() not sum().

    Bucketing narrows each tile to its own rank, but the buckets were launched
    back to back on one stream: a bucket holding 17 of 128 tokens still walks
    the whole K dimension, ~6 us of latency it cannot fill, and four of those in
    series cost more than one padded kernel that does 2x the traffic.  Forking
    the buckets onto separate streams turns that sum into a maximum.  Captured
    inside a CUDA graph these become parallel branches, so the fork costs
    nothing at replay.

    ``streams`` and ``events`` must be allocated by the caller and reused:
    creating them inside a graph capture is not allowed.
    """
    t = x.shape[0]
    out = torch.zeros((t, r_max), device=x.device, dtype=torch.float32)
    main = torch.cuda.current_stream()
    fork = events[0]
    fork.record(main)
    used = []
    for i, (rank, (a_stack, pos, local)) in enumerate(buckets.items()):
        n = pos.shape[0]
        if n == 0:
            continue
        stream = streams[i % len(streams)]
        stream.wait_event(fork)
        with torch.cuda.stream(stream):
            _bucket_shrink_kernel[(n, split_k)](
                x, a_stack, pos, local, out, n, x.shape[1], rank,
                x.stride(0), x.stride(1),
                a_stack.stride(0), a_stack.stride(1), a_stack.stride(2),
                out.stride(0), out.stride(1),
                SPLIT_K=split_k, BK=bk, BR=rank_tile(rank), num_warps=4,
            )
        done = events[i + 1]
        done.record(stream)
        used.append(done)
    for done in used:
        main.wait_event(done)
    return out


@triton.jit
def _bucket_expand_kernel(
    y_ptr, z_ptr, b_ptr, pos_ptr, local_ptr, out_ptr,
    N, R,
    stride_ym, stride_yn,
    stride_zm, stride_zr,
    stride_bn, stride_bo, stride_br,
    stride_om, stride_on,
    scale,
    BN: tl.constexpr, BR: tl.constexpr,
):
    """out[t] = y[t] + scale * z[t, :R] @ B[a, :, :R].T for one rank bucket.

    Same shape argument as the shrink: B is [n_out, R] per adapter, so a uniform
    kernel padded to the batch's largest rank reads r_max columns for every
    token whatever its own rank is.  Compiling per bucket keeps BR at the
    bucket's own rank.
    """
    i = tl.program_id(0)
    pid_n = tl.program_id(1)
    t = tl.load(pos_ptr + i)
    adapter = tl.load(local_ptr + i)

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_r = tl.arange(0, BR)
    mask_n = offs_n < N
    mask_r = offs_r < R

    zv = tl.load(z_ptr + t * stride_zm + offs_r * stride_zr, mask=mask_r, other=0.0)
    bv = tl.load(
        b_ptr + adapter * stride_bn + offs_n[:, None] * stride_bo + offs_r[None, :] * stride_br,
        mask=mask_n[:, None] & mask_r[None, :], other=0.0,
    )
    acc = tl.sum(bv * zv[None, :].to(bv.dtype), axis=1) * scale
    acc += tl.load(y_ptr + t * stride_ym + offs_n * stride_yn, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * stride_om + offs_n * stride_on,
             acc.to(out_ptr.dtype.element_ty), mask=mask_n)


def expand_padded(y, z, b_stack, idx, scale=1.0, bn=128):
    """Every adapter padded to the stack's rank."""
    t, n = y.shape
    r = b_stack.shape[2]
    out = torch.empty_like(y)
    pos = torch.arange(t, device=y.device)
    _bucket_expand_kernel[(t, triton.cdiv(n, bn))](
        y, z, b_stack, pos, idx, out, n, r,
        y.stride(0), y.stride(1), z.stride(0), z.stride(1),
        b_stack.stride(0), b_stack.stride(1), b_stack.stride(2),
        out.stride(0), out.stride(1), scale,
        BN=bn, BR=rank_tile(r), num_warps=4,
    )
    return out


def expand_bucketed_parallel(y, z, buckets_b, streams, events, scale=1.0, bn=128):
    """Rank buckets on concurrent streams, as for the shrink."""
    t, n = y.shape
    out = torch.empty_like(y)
    main = torch.cuda.current_stream()
    fork = events[0]
    fork.record(main)
    used = []
    for i, (rank, (b_stack, pos, local)) in enumerate(buckets_b.items()):
        npos = pos.shape[0]
        if npos == 0:
            continue
        stream = streams[i % len(streams)]
        stream.wait_event(fork)
        with torch.cuda.stream(stream):
            _bucket_expand_kernel[(npos, triton.cdiv(n, bn))](
                y, z, b_stack, pos, local, out, n, rank,
                y.stride(0), y.stride(1), z.stride(0), z.stride(1),
                b_stack.stride(0), b_stack.stride(1), b_stack.stride(2),
                out.stride(0), out.stride(1), scale,
                BN=bn, BR=rank_tile(rank), num_warps=4,
            )
        done = events[i + 1]
        done.record(stream)
        used.append(done)
    for done in used:
        main.wait_event(done)
    return out
