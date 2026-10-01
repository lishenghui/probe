"""Where does FlashTSQR's speed-up come from: real batch parallelism, or a better
single-matrix schedule? Sweep the batch size M at a fixed shape and compare the
PER-MATRIX cost of cuSOLVER vs ours.

  cuSOLVER flat in M          -> it does not exploit the batch at all
  ours falling with M         -> we do
  ours < cuSOLVER even at M=1 -> part of the win is the schedule, not the batch
"""
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline

KERNELS = pathlib.Path(__file__).resolve().parent.parent / "kernels"
ext = load_inline(name="tsqr_bvs", cpp_sources="torch::Tensor tsqr_r(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);",
                  cuda_sources=(KERNELS / "tsqr_r.cu").read_text(),
                  functions=["tsqr_r"], verbose=False, extra_cuda_cflags=["-O3"])


def ev(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.mean(ts)


print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cuda.matmul.allow_tf32 = False
d, N, RPL = 3072, 64, 96
print(f"fixed shape [d={d}, N={N}], sweeping the batch size M\n")
print(f"{'M':>5} | {'cuSOLVER total':>14} {'per-matrix':>11} | {'ours total':>11} {'per-matrix':>11} | {'speedup':>8}")

base1 = None
for M in (1, 2, 4, 8, 16, 32, 56, 112):
    B = torch.randn(M, d, N, device="cuda")
    t_ref = ev(lambda: torch.geqrf(B))
    t_our = ev(lambda: ext.tsqr_r(B, RPL, 256))
    print(f"{M:>5} | {t_ref:>11.3f} ms {t_ref/M:>10.4f} | {t_our:>8.3f} ms {t_our/M:>10.4f} | "
          f"{t_ref/t_our:>7.2f}x")
    del B
    torch.cuda.empty_cache()

print("\nReading:")
print("  cuSOLVER per-matrix roughly CONSTANT in M  -> it never exploits the batch")
print("  ours per-matrix FALLING with M             -> real batch parallelism")
print("  the M=1 column isolates the schedule alone (no batch to exploit)")
