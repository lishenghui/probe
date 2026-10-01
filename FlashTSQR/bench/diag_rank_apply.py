"""How does the full operator's advantage change with the applied rank k?
Our earlier 6.2-18.6x was measured at k=8 only, but real FraQ runs keep
k = mean 36 / max 48 (homo tau=0.95) and ~105 (hetero); merging targets 16-64.

Compares, for k in {8,16,32,N}:
  (a) ours: tsqr_factor + applyQ(k columns)
  (b) cuSOLVER implicit: geqrf + ormqr on a padded [m,k] block
  (c) cuSOLVER explicit: geqrf + orgqr (form Q) + GEMM Q@v
"""
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline

KERNELS = pathlib.Path(__file__).resolve().parent.parent / "kernels"
CPP = ("torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);\n"
       "torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb);")
ext = load_inline(name="tsqr_rksweep", cpp_sources=CPP,
                  cuda_sources=(KERNELS / "tsqr_full.cu").read_text(),
                  functions=["tsqr_factor", "tsqr_applyQ"], verbose=False,
                  extra_cuda_cflags=["-O3"])


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

for (M, m, N, rpl) in [(56, 8192, 64, 128), (56, 3072, 64, 96), (28, 4096, 128, 128)]:
    torch.manual_seed(0)
    B = torch.randn(M, m, N, device="cuda")
    print(f"\n### [{M},{m},{N}]")
    print(f"{'k':>5} | {'ours (fact+apply)':>18} | {'geqrf+ormqr':>12} | {'geqrf+orgqr+GEMM':>17} | "
          f"{'vs ormqr':>8} {'vs orgqr':>8}")
    ks = sorted({8, 16, 32, N})
    for k in ks:
        v = torch.randn(M, N, k, device="cuda")
        vpad = torch.zeros(M, m, k, device="cuda"); vpad[:, :N] = v

        def ours():
            ext.tsqr_factor(B, rpl, 256)
            return ext.tsqr_applyQ(v, 256)

        def implicit():
            a, tau = torch.geqrf(B)
            return torch.ormqr(a, tau, vpad, left=True, transpose=False)

        def explicit():
            Q, R = torch.linalg.qr(B, mode="reduced")   # geqrf + orgqr
            return Q @ v

        t_o = ev(ours); t_i = ev(implicit); t_e = ev(explicit)
        # sanity: ours matches explicit up to column signs
        Qv_o = ours(); Qref, Rref = torch.linalg.qr(B, mode="reduced")
        Rr = ext.tsqr_factor(B, rpl, 256)
        sgn = (torch.sign(torch.diagonal(Rr, dim1=-2, dim2=-1))
               * torch.sign(torch.diagonal(Rref, dim1=-2, dim2=-1)))
        err = ((Qv_o - (Qref * sgn.unsqueeze(1)) @ v).norm() / (Qref @ v).norm()).item()
        print(f"{k:>5} | {t_o:>15.2f} ms | {t_i:>9.2f} ms | {t_e:>14.2f} ms | "
              f"{t_i/t_o:>7.2f}x {t_e/t_o:>7.2f}x   (err {err:.1e})")
        del v, vpad
        torch.cuda.empty_cache()
    del B
    torch.cuda.empty_cache()
