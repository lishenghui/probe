"""Single-launch, unmerged base + LoRA GEMV for very small decode batches.

Each output tile streams W and computes XA locally. Repeating the small shrink
trades rank-dependent work for eliminating a launch and the output round trip.
Unlike the tensor-core GEMM path, ranks below 16 do not round up to 16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_linear_kernel(
    X, W, A, B, Bias, Out,
    K: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
    SXM: tl.constexpr, SXK: tl.constexpr,
    SWN: tl.constexpr, SWK: tl.constexpr,
    SAR: tl.constexpr, SAK: tl.constexpr,
    SBN: tl.constexpr, SBR: tl.constexpr,
    SBI: tl.constexpr,
    SCALE: tl.constexpr, HAS_BIAS: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    rr = tl.arange(0, BR)
    acc = tl.zeros((BN, BK), tl.float32)
    shrink = tl.zeros((BR, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        xv = tl.load(X + m * SXM + k * SXK, k < K, 0).to(tl.float32)
        wv = tl.load(W + n[:, None] * SWN + k[None, :] * SWK,
                     (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
        acc = tl.fma(wv, xv[None, :], acc)
        if R > 0:
            av = tl.load(A + rr[:, None] * SAR + k[None, :] * SAK,
                         (rr[:, None] < R) & (k[None, :] < K), 0).to(tl.float32)
            shrink = tl.fma(av, xv[None, :], shrink)
    result = tl.sum(acc, 1)
    if R > 0:
        # Match the low-precision intermediate used by the cuBLAS sidecar.
        z = tl.sum(shrink, 1).to(X.dtype.element_ty).to(tl.float32)
        bv = tl.load(B + n[:, None] * SBN + rr[None, :] * SBR,
                     (n[:, None] < N) & (rr[None, :] < R), 0).to(tl.float32)
        result += SCALE * tl.sum(bv * z[None, :], 1)
    if HAS_BIAS:
        result += tl.load(Bias + n * SBI, n < N, 0).to(tl.float32)
    tl.store(Out + m * N + n, result, n < N)


def build_decode_linear(weight, bias, a, b, scale=1.0, *, block_n=16, block_k=256):
    """Bind invariant launch metadata once for an inference execution plan.

    The caller validates input dtype/device and rebuilds after factor changes.
    """
    r, k = a.shape
    n = weight.shape[0]
    if r == 0 or scale == 0:
        return lambda x: torch.nn.functional.linear(x, weight, bias)
    strides = (*weight.stride(), *a.stride(), *b.stride())
    rank_tile = triton.next_power_of_2(r)
    scale = float(scale)

    def run(x):
        out = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
        if x.shape[0]:
            _decode_linear_kernel[(x.shape[0], triton.cdiv(n, block_n))](
                x, weight, a, b, bias if bias is not None else x, out,
                k, n, r, *x.stride(), *strides,
                SBI=bias.stride(0) if bias is not None else 0,
                SCALE=scale, HAS_BIAS=bias is not None,
                BN=block_n, BK=block_k, BR=rank_tile, num_warps=4,
            )
        return out

    return run


def decode_lora_linear(x, weight, bias, a, b, scale=1.0, *, block_n=16, block_k=256):
    """Evaluate the complete Linear without packing or changing base weights.

    Inference only. Inputs are 2-D CUDA fp16/bf16 tensors on one device.
    This is a candidate to benchmark, not a universal replacement for cuBLAS.
    """
    r, k = a.shape
    n = weight.shape[0]
    if (x.ndim != 2 or weight.shape[1] != k or x.shape[1] != k
            or b.shape != (n, r) or not 0 <= r <= 64):
        raise ValueError("Incompatible decode Linear shapes or rank outside [0, 64]")
    tensors = (weight, a, b) + (() if bias is None else (bias,))
    if (not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16)
            or any(t.device != x.device or t.dtype != x.dtype for t in tensors)):
        raise ValueError("decode Linear requires CUDA fp16/bf16 tensors of one dtype/device")
    if r == 0 or scale == 0:
        return torch.nn.functional.linear(x, weight, bias)
    if bias is not None and bias.shape != (n,):
        raise ValueError("bias must be a vector of out_features elements")
    return build_decode_linear(weight, bias, a, b, scale, block_n=block_n, block_k=block_k)(x)
