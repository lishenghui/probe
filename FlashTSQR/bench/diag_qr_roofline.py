"""How inefficient is cuSOLVER's batched tall-skinny QR at the LoRA shapes?
Measures achieved GFLOP/s for torch.linalg.qr vs a same-flop GEMM and vs the
CholeskyQR path, at the real Llama/RoBERTa module shapes."""
import statistics
import time

import torch

DEV = "cuda"


def timeit(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.mean(ts)


def qr_flops(M, m, n):      # Householder QR, tall-skinny: ~2mn^2 - 2/3 n^3
    return M * (2 * m * n * n - (2 / 3) * n ** 3)


def cholqr_flops(M, m, n):  # B^T B (2mn^2) + chol (n^3/3) + triangular solve (mn^2)
    return M * (3 * m * n * n + n ** 3 / 3)


print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cuda.matmul.allow_tf32 = False   # fp32 math, fair comparison

# (batch, rows, cols) -- the real shapes: Llama gate/up, attn, k/v, down; RoBERTa
CASES = [
    ("Llama gate/up  [56, 8192, 64]", 56, 8192, 64),
    ("Llama attn     [56, 3072, 64]", 56, 3072, 64),
    ("Llama k/v      [56, 1024, 64]", 56, 1024, 64),
    ("Llama hetero   [56, 8192, 232]", 56, 8192, 232),
    ("RoBERTa q/k/v  [36, 768, 80]", 36, 768, 80),
]

print(f"\n{'case':<32} {'QR ms':>8} {'QR GF/s':>9} | {'CholQR ms':>10} {'CholQR GF/s':>12} | "
      f"{'GEMM ms':>8} {'GEMM GF/s':>10} | {'QR vs GEMM':>10}")
for label, M, m, n in CASES:
    B = torch.randn(M, m, n, device=DEV)

    t_qr = timeit(lambda: torch.linalg.qr(B, mode="reduced"))
    gf_qr = qr_flops(M, m, n) / (t_qr * 1e-3) / 1e9

    def cholqr():
        S = B.mT @ B
        R = torch.linalg.cholesky(S + 1e-6 * torch.eye(n, device=DEV)).mT
        return torch.linalg.solve_triangular(R, B, upper=True, left=False)
    t_ch = timeit(cholqr)
    gf_ch = cholqr_flops(M, m, n) / (t_ch * 1e-3) / 1e9

    # a same-shape GEMM (B^T B) as the "what good looks like" reference
    t_gm = timeit(lambda: B.mT @ B)
    gf_gm = (M * 2 * m * n * n) / (t_gm * 1e-3) / 1e9

    print(f"{label:<32} {t_qr:>8.2f} {gf_qr:>9.1f} | {t_ch:>10.2f} {gf_ch:>12.1f} | "
          f"{t_gm:>8.2f} {gf_gm:>10.1f} | {t_qr/t_gm:>9.1f}x")
    del B
    torch.cuda.empty_cache()

# peak reference: a big square fp32 GEMM
X = torch.randn(8192, 8192, device=DEV)
t = timeit(lambda: X @ X)
print(f"\nreference peak fp32 GEMM (8192^3): {2*8192**3/(t*1e-3)/1e9:.0f} GFLOP/s")
