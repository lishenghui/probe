"""Batched TSQR prototype: shared-memory Householder QR on row blocks + binary
tree reduction. Computes R for a batch of tall-skinny matrices [M, m, N].

Target: beat cuSOLVER's geqrf, which we measured at 84 GF/s (0.17% of a 48 TFLOP/s
GH200) on the real LoRA shapes -- while staying a true Householder QR (unlike the
Cholesky/Gram shortcuts, which crash on rank-deficient B).
"""
import pathlib
import statistics
import time

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = (pathlib.Path(__file__).resolve().parent.parent / "kernels" / "tsqr_r.cu").read_text()

CPP_SRC = "torch::Tensor tsqr_r(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);"

print("compiling CUDA extension ...", flush=True)
ext = load_inline(name="tsqr_ext", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                  functions=["tsqr_r"], verbose=False,
                  extra_cuda_cflags=["-O3", "--use_fast_math"])
print("compiled.\n", flush=True)


def timeit(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.mean(ts)


def qr_flops(M, m, n):
    return M * (2 * m * n * n - (2 / 3) * n ** 3)


print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cuda.matmul.allow_tf32 = False

CASES = [
    ("Llama gate/up [56, 8192, 64]", 56, 8192, 64, 128),
    ("Llama attn    [56, 3072, 64]", 56, 3072, 64, 96),
    ("RoBERTa q/k/v [36, 768, 64]",  36, 768, 64, 96),
]

print(f"\n{'case':<30} {'cuSOLVER geqrf':>16} {'TSQR (ours)':>14} {'speedup':>8} {'|R| err':>10}")
for label, M, m, N, rpl in CASES:
    B = torch.randn(M, m, N, device="cuda")
    F = qr_flops(M, m, N)

    # correctness: |R| must match torch.linalg.qr's |R| (signs may differ)
    R_ref = torch.linalg.qr(B, mode="reduced")[1]
    R_ours = ext.tsqr_r(B, rpl, 256)
    err = ((R_ours.abs() - R_ref.abs()).norm() / R_ref.norm()).item()

    t_ref = timeit(lambda: torch.geqrf(B))
    t_ours = timeit(lambda: ext.tsqr_r(B, rpl, 256))
    print(f"{label:<30} {t_ref:>8.2f} ms {F/(t_ref*1e-3)/1e9:>6.0f}GF "
          f"{t_ours:>7.2f} ms {F/(t_ours*1e-3)/1e9:>5.0f}GF {t_ref/t_ours:>7.2f}x {err:>10.2e}")
    del B, R_ref, R_ours
    torch.cuda.empty_cache()
