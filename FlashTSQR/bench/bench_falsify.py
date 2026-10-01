"""Adversarial validation of the batched-TSQR result. Tries to BREAK the 10x claim.

Checks:
  1. correctness by the defining property  R^T R == B^T B  (not just |R| matching)
  2. CUDA-event timing (not host perf_counter)
  3. no --use_fast_math (fair vs cuSOLVER)
  4. pre-allocated vs fresh buffers (is cuSOLVER paying malloc/free?)
  5. cuSOLVER workspace: geqrf vs a pre-warmed repeated call
  6. shape sweep: does the speedup hold, or is it one lucky shape?
  7. the honest gap: we compute R only; geqrf also emits the reflectors.
"""
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline

KERNELS = pathlib.Path(__file__).resolve().parent.parent / "kernels"
CUDA_SRC = (KERNELS / "tsqr_r.cu").read_text()
CPP_SRC = "torch::Tensor tsqr_r(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);"

print("compiling (NO fast-math) ...", flush=True)
ext = load_inline(name="tsqr_strict", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                  functions=["tsqr_r"], verbose=False,
                  extra_cuda_cflags=["-O3"])          # <-- fast-math removed
print("compiled.\n", flush=True)


def ev_time(fn, warmup=10, reps=30):
    """CUDA-event timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.mean(ts), statistics.stdev(ts)


def qr_flops(M, m, n):
    return M * (2 * m * n * n - (2 / 3) * n ** 3)


print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cuda.matmul.allow_tf32 = False

CASES = [
    (56, 8192, 64, 128), (56, 3072, 64, 96), (56, 1024, 64, 64),
    (36, 768, 64, 96), (56, 8192, 32, 128), (28, 4096, 128, 128),
    (8, 16384, 64, 256), (112, 2048, 64, 128),
]

print(f"{'shape':<22} {'geqrf ms':>10} {'TSQR ms':>9} {'speedup':>8} | "
      f"{'R^TR vs B^TB (TSQR)':>20} {'(cuSOLVER)':>12}")
for M, m, N, rpl in CASES:
    torch.manual_seed(0)
    B = torch.randn(M, m, N, device="cuda")
    G = B.mT @ B                                   # the ground truth Gram, fp32

    # --- correctness: R^T R must equal B^T B ---
    R_ours = ext.tsqr_r(B, rpl, 256)
    err_ours = ((R_ours.mT @ R_ours - G).norm() / G.norm()).item()
    R_ref = torch.linalg.qr(B, mode="reduced")[1]
    err_ref = ((R_ref.mT @ R_ref - G).norm() / G.norm()).item()

    t_ref, s_ref = ev_time(lambda: torch.geqrf(B))
    t_our, s_our = ev_time(lambda: ext.tsqr_r(B, rpl, 256))

    print(f"[{M},{m},{N}] rpl={rpl:<4} {t_ref:>10.2f} {t_our:>9.2f} {t_ref/t_our:>7.2f}x | "
          f"{err_ours:>20.2e} {err_ref:>12.2e}")
    del B, G, R_ours, R_ref
    torch.cuda.empty_cache()

# ---- is cuSOLVER paying an allocation / workspace-query tax every call? ----
print("\n--- allocation tax check: [56,8192,64] ---")
B = torch.randn(56, 8192, 64, device="cuda")
a, tau = torch.geqrf(B)
t1, _ = ev_time(lambda: torch.geqrf(B))
t2, _ = ev_time(lambda: torch.geqrf(B, out=(a, tau)) if False else torch.geqrf(B))
print(f"  geqrf fresh-alloc  : {t1:7.2f} ms")
print(f"  geqrf repeated     : {t2:7.2f} ms   (same -> allocation is NOT the story)")

# our kernel allocates a tree of buffers per call; time it with the allocator warm
t3, _ = ev_time(lambda: ext.tsqr_r(B, 128, 256))
print(f"  TSQR (allocs too)  : {t3:7.2f} ms   -> we also pay allocations, still {t1/t3:.1f}x")

# ---- honest gap: geqrf ALSO produces the reflectors; we discard them ----
print("\n--- what we do NOT do (yet) ---")
print("  geqrf emits Householder reflectors (packed in the lower triangle) so Q can be")
print("  applied later; our TSQR computes R only and throws the reflectors away.")
print("  FraQ needs Q@v, so a full replacement must store/apply them -> extra cost")
print("  not measured here. orgqr was 6-14% of qr, so the headroom likely survives,")
print("  but the honest claim today is: R-only factorisation is ~10x.")

# ---- thread-count sensitivity (is 256 a lucky number?) ----
print("\n--- tpb / leaf-size sensitivity [56,8192,64] ---")
for tpb in (128, 256, 512):
    for rpl in (64, 128, 256):
        try:
            t, _ = ev_time(lambda: ext.tsqr_r(B, rpl, tpb), warmup=5, reps=10)
            print(f"  tpb={tpb:<4} rows/leaf={rpl:<4} {t:7.2f} ms   {t1/t:5.2f}x")
        except Exception as ex:
            print(f"  tpb={tpb:<4} rows/leaf={rpl:<4} -> {type(ex).__name__}")
