"""Rank-proportional LoRA linear layers for sidecar inference.

Why this exists
---------------
A sidecar LoRA is normally evaluated as three separate passes over the
activations::

    z      = x @ A.T                  # reads all of x        (M*K elements)
    update = z @ B.T                  # writes all of update  (M*N elements)
    out    = base + scale * update    # reads base, reads update, writes out

Only the ``M*r`` intermediate depends on the rank.  Every other term is
rank-independent, and once ``r`` is small both GEMMs are memory bound, so
truncating a rank-256 adapter to rank 8 removes flops that were never the
bottleneck.  That is why the measured e100/e95/e90/e80/e70 latency curve is
flat even though FraQ cuts the mean rank by more than 20x.

The implementations here
------------------------
``concat``
    Pack ``A`` into the base weight as extra output rows, so one cuBLAS GEMM
    reads ``x`` once and emits ``[y | z]`` together.  The GEMM then costs
    exactly ``(N + r) / N`` of the un-adapted layer -- proportional to the rank,
    at cuBLAS efficiency.  A Triton epilogue folds ``y + scale * z @ B.T`` into
    a single pass.  This is the default for shapes where the base GEMM is
    compute bound.

``fused``
    One Triton kernel walks K once and feeds the resident ``[BM, BK]``
    activation tile to both the base and the shrink accumulator, then folds the
    expand into the epilogue.  The sidecar costs *no* extra memory traffic at
    all, which wins on the memory-bound shapes (small K and N, huge M) that
    dominate video-diffusion temporal attention.  It needs the whole rank live
    in registers, so it is capped at ``FUSED_MAX_RANK``.

``hybrid``
    cuBLAS for both GEMMs plus the Triton epilogue.  Needs no weight packing, so
    this is what the PEFT forward patch uses for adapters it must not rewrite.

``torch``
    The naive three-pass formulation, kept so the autoselector can never lose to
    the baseline on a shape none of the fast paths suit.

:func:`select_variant` measures the legal candidates once per shape and caches
the winner.  Diffusion and decode workloads replay a handful of shapes thousands
of times, amortising that measurement. This is a per-shape heuristic: cache
behaviour and host scheduling still require full-model validation.
"""

from __future__ import annotations

import os
import statistics

import torch
import triton
import triton.language as tl

from .decode_linear import build_decode_linear

# Triton block shapes must be powers of two and a tensor-core tile needs at
# least 16 along the contracted dimension, so the rank tile is the next power of
# two at or above 16.  That still gives distinct cost tiers at 16/32/64/128
# across the range FraQ produces, versus the single flat tier today.
MIN_RANK_TILE = 16

# Above this the [BM, BR] shrink accumulator stops fitting in registers
# alongside the [BM, BN] output accumulator.
FUSED_MAX_RANK = 64

# Below this many rows a GEMM-shaped kernel is pure overhead.
MIN_GEMM_ROWS = 128

VARIANTS = ("kconcat", "concat", "fused", "hybrid", "torch", "decode8", "decode16", "decode32")


def pad_rank(r: int) -> int:
    """Round a rank up to the next legal tensor-core tile width."""
    return max(MIN_RANK_TILE, triton.next_power_of_2(r))


def _dot_precision(dtype) -> str:
    """Match torch's fp32 matmul precision so results stay comparable."""
    if dtype is not torch.float32:
        return "tf32"
    return "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"


# --------------------------------------------------------------------------- #
# Epilogue: out = y + scale * z @ B.T
# --------------------------------------------------------------------------- #

def _epilogue_configs():
    return [
        triton.Config({"BM": 128, "BN": 128}, num_stages=3, num_warps=8),
        triton.Config({"BM": 64, "BN": 256}, num_stages=3, num_warps=8),
        triton.Config({"BM": 256, "BN": 64}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 64}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 64}, num_stages=4, num_warps=4),
        triton.Config({"BM": 256, "BN": 128}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 256}, num_stages=3, num_warps=8),
        triton.Config({"BM": 256, "BN": 32}, num_stages=4, num_warps=4),
        triton.Config({"BM": 512, "BN": 64}, num_stages=3, num_warps=8),
    ]


# `out` never aliases `y`, which matters more than the allocation it costs: with
# aliasing the autotuner needs restore_value, and saving/restoring a 100 MB
# output between candidates swamps the kernel it is trying to time.  Measured,
# that mis-picked the config by 20% at N=1536 and 50% at N=5120.  The kernel
# writes out either way, so a fresh buffer adds no traffic.
@triton.autotune(configs=_epilogue_configs(), key=["M", "N", "BR"])
# EVEN has to be decided after the config is chosen, since it depends on the
# tile sizes, so it is a heuristic rather than a caller-supplied constexpr.
@triton.heuristics({
    "EVEN": lambda a: a["M"] % a["BM"] == 0 and a["N"] % a["BN"] == 0 and a["R"] % a["BR"] == 0,
})
@triton.jit
def _expand_add_kernel(
    y_ptr, z_ptr, b_ptr, out_ptr,
    M, N, R,
    stride_ym, stride_yn,
    stride_zm, stride_zr,
    stride_bn, stride_br,
    stride_om, stride_on,
    scale,
    BM: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
    PRECISION: tl.constexpr, EVEN: tl.constexpr,
):
    """out = y + scale * z @ B.T in one pass over the [M, N] activation.

    ``EVEN`` drops every mask when the tiles divide the problem exactly.  Masked
    loads cannot be widened, and this kernel is bandwidth bound, so the
    unmasked specialisation is worth the extra compile.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_r = tl.arange(0, BR)
    mask_m = offs_m < M
    mask_n = offs_n < N

    z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_r[None, :] * stride_zr
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_r[:, None] * stride_br

    # The rank is this GEMM's contracted dimension, so walk it in tiles: nothing
    # has to stay resident and arbitrarily large ranks work.
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for r0 in range(0, tl.cdiv(R, BR)):
        if EVEN:
            z_tile = tl.load(z_ptrs)
            b_tile = tl.load(b_ptrs)
        else:
            mask_r = offs_r < R - r0 * BR
            z_tile = tl.load(z_ptrs, mask=mask_m[:, None] & mask_r[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(z_tile, b_tile, acc, input_precision=PRECISION)
        z_ptrs += BR * stride_zr
        b_ptrs += BR * stride_br

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if EVEN:
        acc = acc * scale + tl.load(y_ptrs).to(tl.float32)
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))
    else:
        mask_out = mask_m[:, None] & mask_n[None, :]
        acc = acc * scale + tl.load(y_ptrs, mask=mask_out, other=0.0).to(tl.float32)
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_out)


def expand_add(y, z, b, scale):
    """``out = y + scale * z @ b.T`` for y [M,N], z [M,R], b [N,R].

    ``y`` and ``z`` may be strided views -- that is how the ``concat`` path
    slices one GEMM result into two, and it costs nothing measurable.  The
    output is always a fresh buffer; see the autotune note above.
    """
    m, r = z.shape
    n = b.shape[0]
    assert b.shape[1] == r and y.shape == (m, n)
    out = torch.empty((m, n), device=y.device, dtype=y.dtype)
    br = min(pad_rank(r), 64)
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    _expand_add_kernel[grid](
        y, z, b, out,
        m, n, r,
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        scale,
        BR=br,
        PRECISION=_dot_precision(y.dtype),
    )
    return out


# --------------------------------------------------------------------------- #
# Single-kernel fused GEMM
# --------------------------------------------------------------------------- #

def _fused_configs():
    # Deliberately conservative on BN: the [BM, BR] shrink accumulator shares the
    # register file with the [BM, BN] output accumulator, and spilling there
    # costs far more than the wider tile buys.
    return [
        # Wide-BN tiles matter more than they look: the grid re-reads the whole
        # activation once per column block, so a layer with N=320 and BN=64 pays
        # five passes over x before any sidecar work happens.
        triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 256, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BM": 64, "BN": 256, "BK": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 256, "BN": 128, "BK": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 64, "BK": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BM": 64, "BN": 64, "BK": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 256, "BN": 64, "BK": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 64, "BN": 64, "BK": 64, "GROUP_M": 8}, num_stages=5, num_warps=4),
    ]


@triton.autotune(configs=_fused_configs(), key=["M", "N", "K", "BR"])
@triton.jit
def _fused_lora_gemm_kernel(
    x_ptr, w_ptr, bias_ptr, a_ptr, b_ptr, out_ptr,
    M, N, K, R,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ar, stride_ak,
    stride_bn, stride_br,
    stride_om, stride_on,
    scale,
    HAS_BIAS: tl.constexpr, HAS_LORA: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """out = x @ W.T + bias + scale * (x @ A.T) @ B.T, in one pass over K."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    # Grouped launch order keeps the reused W tiles hot in L2.
    pids_per_group = GROUP_M * num_pid_n
    group_id = pid // pids_per_group
    first_pid_m = group_id * GROUP_M
    group_rows = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % pids_per_group) % group_rows)
    pid_n = (pid % pids_per_group) // group_rows

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_r = tl.arange(0, BR)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_r = offs_r < R

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    a_ptrs = a_ptr + offs_r[None, :] * stride_ar + offs_k[:, None] * stride_ak

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    z = tl.zeros((BM, BR), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        mask_k = offs_k < K - k0 * BK
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(x_tile, w_tile, acc)
        if HAS_LORA:
            # The shrink rides on the activation tile that is already resident,
            # so it costs flops but no additional global traffic.
            a_tile = tl.load(a_ptrs, mask=mask_k[:, None] & mask_r[None, :], other=0.0)
            z = tl.dot(x_tile, a_tile, z)
            a_ptrs += BK * stride_ak
        x_ptrs += BK * stride_xk
        w_ptrs += BK * stride_wk

    if HAS_LORA:
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_r[:, None] * stride_br
        b_tile = tl.load(b_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0)
        acc += scale * tl.dot(z.to(b_tile.dtype), b_tile)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]

    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def fused_gemm(x, weight, bias, a, b, scale, out=None):
    """Single-kernel ``x @ weight.T + bias + scale * (x @ a.T) @ b.T``."""
    m = x.shape[0]
    n = weight.shape[0]
    r = 0 if a is None else a.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    _fused_lora_gemm_kernel[grid](
        x, weight, bias if bias is not None else x, a if a is not None else x,
        b if b is not None else x, out,
        m, n, x.shape[1], r,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        a.stride(0) if a is not None else 0, a.stride(1) if a is not None else 0,
        b.stride(0) if b is not None else 0, b.stride(1) if b is not None else 0,
        out.stride(0), out.stride(1),
        scale,
        HAS_BIAS=bias is not None, HAS_LORA=a is not None,
        BR=pad_rank(r) if a is not None else MIN_RANK_TILE,
    )
    return out


# --------------------------------------------------------------------------- #
# Variant implementations
# --------------------------------------------------------------------------- #

# cuBLAS drops off its tensor-core kernels when the output dimension is not a
# multiple of 8 (16 for the widest tiles).  Packing a rank-17 sidecar onto a
# 320-wide layer would make N + r == 337 and cost more than the sidecar saves,
# so the packed block is padded with zero rows up to this alignment.
PACK_ALIGN = 16


def packed_rank(r: int) -> int:
    """Rank rows actually appended, rounded up to keep the GEMM tile aligned."""
    return ((r + PACK_ALIGN - 1) // PACK_ALIGN) * PACK_ALIGN


def augment_weight(weight: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Pack the shrink factor into the base weight as extra output rows."""
    a = a.to(device=weight.device, dtype=weight.dtype)
    pad = packed_rank(a.shape[0]) - a.shape[0]
    blocks = [weight, a]
    if pad:
        blocks.append(a.new_zeros(pad, a.shape[1]))
    return torch.cat(blocks, dim=0).contiguous()


def augment_bias(bias: torch.Tensor | None, rank: int) -> torch.Tensor | None:
    """Extend a bias with zeros so the packed rows stay unbiased."""
    if bias is None:
        return None
    return torch.cat([bias, bias.new_zeros(packed_rank(rank))]).contiguous()


# Timing-only escape hatch.  With this set the expand is skipped entirely, so the
# result is WRONG.  Treat the number it produces with suspicion too: the early
# return hands back a non-contiguous view of the packed GEMM output, and the
# extra cost that imposes on downstream ops is itself rank-independent, so it
# manufactures exactly the kind of flat floor one would be hunting for.  Never
# set it for real inference, and prefer a profile over this.
_SKIP_EPILOGUE = os.environ.get("LORAFORGE_SKIP_EPILOGUE", "0") == "1"


def _concat(x, weight_aug, bias_aug, b, scale, n):
    y_aug = torch.nn.functional.linear(x, weight_aug, bias_aug)
    if _SKIP_EPILOGUE:
        return y_aug[:, :n]
    # Slice off the alignment padding; b carries the true rank.
    return expand_add(y_aug[:, :n], y_aug[:, n : n + b.shape[1]], b, scale)


def _hybrid(x, weight, bias, a, b, scale):
    y = torch.nn.functional.linear(x, weight, bias)
    z = torch.mm(x, a.t())
    return expand_add(y, z, b, scale)


def _torch_naive(x, weight, bias, a, b, scale):
    y = torch.nn.functional.linear(x, weight, bias)
    return y + scale * torch.mm(torch.mm(x, a.t()), b.t())


def _half(dtype):
    return dtype in (torch.float16, torch.bfloat16)


def eligible(variant, m, k, n, r, dtype, has_augmented):
    # concat and hybrid keep cuBLAS for every GEMM, so they are safe in fp32 too;
    # only the Triton epilogue changes, and it matches torch's fp32 precision.
    if variant == "concat":
        return has_augmented and m >= MIN_GEMM_ROWS
    if variant == "fused":
        # Measured: with a 16-wide rank tile the fused kernel runs 2-5x slower
        # than every alternative on every shape tried, and it never wins there.
        # The cause is not established; it is excluded rather than chased,
        # because concat already handles that end of the rank range well.
        return (_half(dtype) and m >= MIN_GEMM_ROWS
                and MIN_RANK_TILE < pad_rank(r) <= FUSED_MAX_RANK)
    if variant == "kconcat":
        # kconcat rewrites the activation (3*M*K) where concat re-reads the
        # output (2*M*N), so it only pays when 3K < 2N.  Gate on that rather
        # than let the probe build a packed weight it will discard: measured,
        # it wins on an FFN up projection (K=1536, N=8960: 50.5% -> 28.3%
        # sidecar overhead at rank 256) and loses badly on an FFN down.
        return has_augmented and _half(dtype) and m >= MIN_GEMM_ROWS and 3 * k < 2 * n
    if variant == "hybrid":
        return m >= MIN_GEMM_ROWS
    return True


# --------------------------------------------------------------------------- #
# Variant selection
# --------------------------------------------------------------------------- #

def _time_ms(fn, warmup=5, reps=15, use_graph=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    inner = 1
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        inner = 20
        with torch.cuda.graph(graph):
            for _ in range(inner):
                fn()
        fn = graph.replay
    samples = []
    for _ in range(reps):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / inner)
    return statistics.median(samples)


_SELECTION: dict[tuple, str] = {}
_SELECTION_CONTEXT: dict[tuple, tuple] = {}


def _forced_variant() -> str | None:
    choice = os.environ.get("LORAFORGE_VARIANT", "").strip().lower()
    return choice if choice in VARIANTS else None


def select_variant(runners: dict, key: tuple) -> str:
    """Time each legal implementation once for this shape and cache the winner."""
    forced = _forced_variant()
    if forced is not None and forced in runners:
        return forced
    cached = _SELECTION.get(key)
    context = (tuple(runners), os.environ.get("LORAFORGE_TUNE_MODE", "eager"))
    if cached in runners and _SELECTION_CONTEXT.get(key) == context:
        return cached
    timings = {}
    for variant, fn in runners.items():
        try:
            timings[variant] = _time_ms(
                fn, use_graph=key[0] <= 8 and os.environ.get("LORAFORGE_TUNE_MODE", "eager") == "graph",
            )
        except Exception:  # a variant that fails to compile simply drops out
            continue
    best = min(timings, key=timings.get) if timings else "torch"
    _SELECTION[key] = best
    _SELECTION_CONTEXT[key] = context
    return best


def selection_report() -> dict:
    """Which variant won for each shape seen so far (for logging / debugging)."""
    return {
        f"m{m}_k{k}_n{n}_r{r}_{str(dtype).split('.')[-1]}": variant
        for (m, k, n, r, dtype, _dev), variant in _SELECTION.items()
    }


def build_plan(probe, weight, bias, a, b, scale, base_layer=None):
    """Resolve the variant for this shape once and return a specialised callable.

    The packed weights the fast paths need are built here, measured, and then
    only the winner's is kept -- the losers' are freed with their closures.  The
    per-call work in the generic entry point is small in absolute terms but a
    diffusion step calls it once per adapted linear per CFG pass, thousands of
    times per generation, so a plan is resolved once and then invoked directly.

    ``base_layer`` enables the memory optimisation for the ``concat`` winner:
    its weight is rebound to the leading rows of the packed buffer, so the layer
    grows by ``rank * in_features`` instead of holding a second copy.
    """
    n, k = weight.shape
    r = a.shape[0]
    dtype = a.dtype
    rows = probe.shape[0]
    if r == 0 or scale == 0:
        return lambda x: torch.nn.functional.linear(x, weight, bias)
    pack = os.environ.get("LORAFORGE_PACK", "view").strip().lower()
    allow_pack = pack not in ("0", "off", "none")

    runners = {"torch": lambda x: _torch_naive(x, weight, bias, a, b, scale)}
    if (_half(dtype) and rows <= 8 and r <= 16 and k <= 4096
            and (rows == 1 or n <= 4096)
            and os.environ.get("LORAFORGE_ENABLE_DECODE_LINEAR", "1") == "1"):
        for tile in (8, 16, 32):
            runners[f"decode{tile}"] = build_decode_linear(weight, bias, a, b, scale, block_n=tile)
    if eligible("hybrid", rows, k, n, r, dtype, False):
        runners["hybrid"] = lambda x: _hybrid(x, weight, bias, a, b, scale)
    if eligible("fused", rows, k, n, r, dtype, False):
        runners["fused"] = lambda x: fused_gemm(x, weight, bias, a, b, scale)
    weight_aug = bias_aug = weight_kcat = None
    if allow_pack and eligible("concat", rows, k, n, r, dtype, True):
        weight_aug = augment_weight(weight, a)
        bias_aug = augment_bias(bias, r)
        runners["concat"] = lambda x: _concat(x, weight_aug, bias_aug, b, scale, n)
    if allow_pack and eligible("kconcat", rows, k, n, r, dtype, True):
        weight_kcat = kconcat_weight(weight, b)
        rank_pad = packed_rank(r)
        runners["kconcat"] = lambda x: _kconcat(x, weight_kcat, bias, a, scale, rank_pad)

    key = (rows, k, n, r, dtype, torch.cuda.current_device())
    variant = select_variant({nm: (lambda f=fn: f(probe)) for nm, fn in runners.items()}, key)

    if variant == "concat" and pack != "copy" and base_layer is not None:
        if isinstance(base_layer.weight, torch.nn.Parameter):
            base_layer.weight = torch.nn.Parameter(weight_aug[:n], requires_grad=False)
    return runners[variant]


def fused_lora_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """``x @ weight.T + bias + scale * (x @ a.T) @ b.T`` with the sidecar fused in.

    ``x`` may carry any number of leading dimensions.  ``a`` is ``[r, k]`` and
    ``b`` is ``[n, r]``, matching ``lora_A.weight`` / ``lora_B.weight``.  This
    resolves a plan on every call; hot paths should hold one from
    :func:`build_plan` instead.
    """
    shape = x.shape
    n = weight.shape[0]
    x2 = x.reshape(-1, shape[-1])
    if a.shape[0] == 0 or scale == 0:
        return torch.nn.functional.linear(x, weight, bias)
    if not x2.is_cuda:
        return _torch_naive(x2, weight, bias, a, b, scale).reshape(*shape[:-1], n)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    return build_plan(x2, weight, bias, a, b, scale)(x2).reshape(*shape[:-1], n)


class FusedLoRALinear(torch.nn.Module):
    """Drop-in replacement for an ``nn.Linear`` carrying a LoRA sidecar.

    Nothing is merged into the base weight, so the adapter stays swappable.  The
    packed buffers the fast paths need are built and chosen by
    :func:`build_plan` on first use.
    """

    def __init__(self, base: torch.nn.Linear, a: torch.Tensor, b: torch.Tensor, scale: float = 1.0):
        super().__init__()
        if a.shape[1] != base.in_features or b.shape[0] != base.out_features or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"LoRA factors {tuple(a.shape)}/{tuple(b.shape)} do not match "
                f"Linear({base.in_features}, {base.out_features})"
            )
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.scale = float(scale)
        device, dtype = base.weight.device, base.weight.dtype
        self.register_buffer("weight", base.weight.detach())
        self.register_buffer("bias", None if base.bias is None else base.bias.detach())
        self.register_buffer("lora_a", a.detach().to(device=device, dtype=dtype).contiguous())
        self.register_buffer("lora_b", b.detach().to(device=device, dtype=dtype).contiguous())
        self._plan = None
        self._plan_rows = -1

    @property
    def rank(self) -> int:
        return self.lora_b.shape[1]

    def set_factors(self, a: torch.Tensor, b: torch.Tensor, scale: float | None = None) -> None:
        """Swap the adapter in place; the plan is rebuilt around the new factors."""
        device, dtype = self.weight.device, self.weight.dtype
        self.lora_a = a.detach().to(device=device, dtype=dtype).contiguous()
        self.lora_b = b.detach().to(device=device, dtype=dtype).contiguous()
        if scale is not None:
            self.scale = float(scale)
        self._plan = None
        self._plan_rows = -1

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rank={self.rank}, scale={self.scale}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resolve once per row count and reuse: this runs once per adapted linear
        # per denoising step.
        shape = x.shape
        rows = x.numel() // shape[-1]
        if not x.is_cuda:
            return fused_lora_linear(x, self.weight, self.bias, self.lora_a,
                                     self.lora_b, self.scale)
        x2 = x.reshape(-1, shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if self._plan_rows != rows:
            self._plan = build_plan(x2, self.weight, self.bias, self.lora_a,
                                    self.lora_b, self.scale)
            self._plan_rows = rows
        return self._plan(x2).reshape(*shape[:-1], self.out_features)


# --------------------------------------------------------------------------- #
# K-concat: turn the whole sidecar into one ordinary GEMM
# --------------------------------------------------------------------------- #
#
#   out = x @ W.T + scale * (x @ A.T) @ B.T
#       = [x | scale*z] @ [W | B].T          with z = x @ A.T
#
# Concatenating along K instead of along N makes the sidecar a plain GEMM that
# cuBLAS runs at full speed, with no epilogue pass over the [M, N] activation at
# all.  The price is building `[x | scale*z]` contiguously, which one Triton
# kernel does in a single pass over x: it copies x into the head of the buffer
# and accumulates the shrink into its tail at the same time.  That kernel is a
# skinny GEMM plus a copy, so it never has to compete with cuBLAS.
#
# Extra traffic is 2*M*K rather than concat's 2*M*N, which is why it wins on
# wide layers (K << N, e.g. an FFN up projection: measured 50.4% -> 22.5%
# sidecar overhead at rank 256) and loses on narrow ones (K >> N, e.g. an FFN
# down projection).  `select_variant` picks between them per shape.


# Building `[x | scale*z]` was first tried as one Triton kernel that copied x
# and accumulated the shrink in a single pass over x (2*M*K of traffic).  It is
# not worth it: two separate Triton 3.7 miscompiles showed up on GH200, both
# silent -- wrong numbers, no error raised, and only for some autotuned tile
# configs, so any single-config test would have missed them.  A [BM, BR]
# accumulator at BR >= 128 corrupted z, and so did a store inside the pipelined
# K loop, whether branched or mask-predicated.
#
# So this builds ext out of plain PyTorch ops instead.  That costs one extra
# pass over x (3*M*K rather than 2*M*K) and buys correctness by construction.
# Even at 3*M*K it is far below concat's 2*M*N whenever K << N, which is the
# case kconcat exists for.


def kconcat_weight(weight: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[W | B] widened along K, zero-padded so K + pad stays GEMM-aligned."""
    n, k = weight.shape
    r = b.shape[1]
    out = weight.new_zeros((n, k + packed_rank(r)))
    out[:, :k] = weight
    out[:, k : k + r] = b.to(device=weight.device, dtype=weight.dtype)
    return out.contiguous()


def build_ext(x, a, scale, rank_pad):
    """Materialise ``[x | scale * x @ a.T]`` contiguously."""
    m, k = x.shape
    r = a.shape[0]
    ext = torch.empty((m, k + rank_pad), device=x.device, dtype=x.dtype)
    ext[:, :k].copy_(x)
    ext[:, k : k + r].copy_(torch.mm(x, a.t()).mul_(scale))
    if rank_pad > r:
        # The matching columns of the packed weight are zero as well, so either
        # side alone would do; zeroing here keeps ext valid for any weight.
        ext[:, k + r :].zero_()
    return ext


def _kconcat(x, weight_kcat, bias, a, scale, rank_pad):
    return torch.nn.functional.linear(build_ext(x, a, scale, rank_pad), weight_kcat, bias)
