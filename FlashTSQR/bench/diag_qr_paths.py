"""Is the batched tall-skinny QR slowness cuSOLVER's fault, or PyTorch's dispatch?
Compares: torch.linalg.qr (geqrf+orgqr) | geqrf alone | orgqr alone | per-matrix loop
| chunked | geqrf+ormqr (apply Q implicitly to a small matrix, no Q materialisation).
"""
import statistics
import time

import torch

DEV = "cuda"


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
RANK = 8   # FraQ only ever needs Q @ v with v = [n, rank]

CASES = [
    ("Llama gate/up [56, 8192, 64]", 56, 8192, 64),
    ("Llama attn    [56, 3072, 64]", 56, 3072, 64),
    ("Llama hetero  [56, 8192, 232]", 56, 8192, 232),
]

for label, M, m, n in CASES:
    B = torch.randn(M, m, n, device=DEV)
    F = qr_flops(M, m, n)
    print(f"\n### {label}   (QR flops = {F/1e9:.1f} GFLOP)")

    def gf(t):
        return F / (t * 1e-3) / 1e9

    t_qr = timeit(lambda: torch.linalg.qr(B, mode="reduced"))
    print(f"  torch.linalg.qr  (geqrf+orgqr)      {t_qr:9.2f} ms   {gf(t_qr):8.1f} GF/s   1.00x")

    t_geqrf = timeit(lambda: torch.geqrf(B))
    print(f"  torch.geqrf      (factor only)      {t_geqrf:9.2f} ms   {gf(t_geqrf):8.1f} GF/s   "
          f"{t_qr/t_geqrf:.2f}x")

    a, tau = torch.geqrf(B)
    t_orgqr = timeit(lambda: torch.linalg.householder_product(a, tau))
    print(f"  orgqr            (form Q only)      {t_orgqr:9.2f} ms   {'':>8}       "
          f"(= {100*t_orgqr/t_qr:.0f}% of qr)")

    # geqrf + ormqr: never materialise Q; apply it straight to a small [n, rank] block
    v = torch.randn(M, m, RANK, device=DEV)   # padded [m, rank] (top n rows = v, rest 0)
    def geqrf_ormqr():
        aa, tt = torch.geqrf(B)
        return torch.ormqr(aa, tt, v, left=True, transpose=False)
    try:
        t_go = timeit(geqrf_ormqr)
        print(f"  geqrf + ormqr    (implicit Q@v)     {t_go:9.2f} ms   {gf(t_go):8.1f} GF/s   "
              f"{t_qr/t_go:.2f}x  <-- no Q materialisation")
    except Exception as e:
        print(f"  geqrf + ormqr    -> {type(e).__name__}: {e}")

    # per-matrix python loop (is the batched path worse than looping?)
    t_loop = timeit(lambda: [torch.linalg.qr(B[i], mode="reduced") for i in range(M)], warmup=2, reps=5)
    print(f"  per-matrix loop  ({M} x linalg.qr)   {t_loop:9.2f} ms   {gf(t_loop):8.1f} GF/s   "
          f"{t_qr/t_loop:.2f}x")

    # chunked batches
    for ch in (8, 16):
        t_ch = timeit(lambda c=ch: [torch.linalg.qr(B[i:i+c], mode="reduced")
                                    for i in range(0, M, c)], warmup=2, reps=5)
        print(f"  chunked batch={ch:<3}                   {t_ch:9.2f} ms   {gf(t_ch):8.1f} GF/s   "
              f"{t_qr/t_ch:.2f}x")

    del B, a, tau, v
    torch.cuda.empty_cache()

X = torch.randn(8192, 8192, device=DEV)
t = timeit(lambda: X @ X)
print(f"\nreference peak fp32 GEMM: {2*8192**3/(t*1e-3)/1e9:.0f} GFLOP/s")
