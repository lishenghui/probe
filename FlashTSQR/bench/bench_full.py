"""Full batched TSQR operator: R + Householder reflectors + Q@v application.
Fair comparison against cuSOLVER's geqrf + ormqr (which is what FraQ actually needs).
Correctness: R^T R == B^T B, and Q@v matches torch's Q@v.
"""
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline

CPP = ("torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);\n"
       "torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb);")

print("compiling ...", flush=True)
ext = load_inline(name="tsqr_full", cpp_sources=CPP,
                  cuda_sources=(pathlib.Path(__file__).resolve().parent.parent / "kernels" / "tsqr_full.cu").read_text(),
                  functions=["tsqr_factor", "tsqr_applyQ"], verbose=False,
                  extra_cuda_cflags=["-O3"])
print("compiled.\n", flush=True)


def ev_time(fn, warmup=10, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.mean(ts), statistics.stdev(ts)


print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cuda.matmul.allow_tf32 = False
RANK = 8

CASES = [(56, 8192, 64, 128), (56, 3072, 64, 96), (56, 1024, 64, 64), (36, 768, 64, 96)]

print(f"{'shape':<22} {'geqrf+ormqr':>12} {'TSQR full':>10} {'speedup':>8} | "
      f"{'R^TR-B^TB':>10} {'Q@v err':>9}")
for M, m, N, rpl in CASES:
    torch.manual_seed(0)
    B = torch.randn(M, m, N, device="cuda")
    v = torch.randn(M, N, RANK, device="cuda")
    G = B.mT @ B

    # ---- ours: factor + applyQ
    R = ext.tsqr_factor(B, rpl, 256)
    Qv = ext.tsqr_applyQ(v, 256)
    err_R = ((R.mT @ R - G).norm() / G.norm()).item()

    # ---- reference Q@v via torch
    Qref, Rref = torch.linalg.qr(B, mode="reduced")
    Qv_ref = Qref @ v
    # QR signs are not unique: align by the sign of R's diagonal
    sgn = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    sgn_ref = torch.sign(torch.diagonal(Rref, dim1=-2, dim2=-1))
    flip = (sgn * sgn_ref).unsqueeze(-1)               # [M,N,1]
    err_Qv = ((Qv - (Qref * (sgn * sgn_ref).unsqueeze(1)) @ v).norm()
              / Qv_ref.norm()).item()

    # ---- baseline: geqrf + ormqr (never materialises Q, same as what FraQ needs)
    vpad = torch.zeros(M, m, RANK, device="cuda")
    vpad[:, :N] = v
    def baseline():
        a, tau = torch.geqrf(B)
        return torch.ormqr(a, tau, vpad, left=True, transpose=False)
    t_b, _ = ev_time(baseline)

    def ours():
        ext.tsqr_factor(B, rpl, 256)
        return ext.tsqr_applyQ(v, 256)
    t_o, _ = ev_time(ours)

    print(f"[{M},{m},{N}] rpl={rpl:<4} {t_b:>12.2f} {t_o:>10.2f} {t_b/t_o:>7.2f}x | "
          f"{err_R:>10.2e} {err_Qv:>9.2e}")
    del B, v, G, R, Qv, Qref, Rref, vpad
    torch.cuda.empty_cache()
