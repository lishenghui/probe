"""Large-N support: arbitrary leaf counts (bye-passing tree) + zero-padded tail
leaf unlock the heterogeneous Sigma_r=232 shape that was previously rejected by
the m % rpl == 0 / power-of-two constraints.

Also regression-checks the old power-of-two shapes through the generalised tree.
k values for N=232 mirror the real hetero run: cap 8, tau=0.95 effective (~105), full.
"""
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline

K = pathlib.Path(__file__).resolve().parent.parent / "kernels"
CPP = ("torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);\n"
       "torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb);")
print("compiling ...", flush=True)
ext = load_inline(name="tsqr_bigN", cpp_sources=CPP,
                  cuda_sources=(K / "tsqr_full.cu").read_text(),
                  functions=["tsqr_factor", "tsqr_applyQ"], verbose=False,
                  extra_cuda_cflags=["-O3"])
print("compiled.\n", flush=True)


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

CASES = [
    # regression: the old power-of-two shapes
    (56, 8192, 64, 128, (8, 64)),
    (28, 4096, 128, 128, (8, 128)),
    # new: real hetero shapes -- P=36 with byes (8192 = 35x232 + 72) and P=14
    (56, 8192, 232, 232, (8, 105, 232)),
    (56, 3072, 232, 232, (8, 105)),
]

for (M, m, N, rpl, ks) in CASES:
    torch.manual_seed(0)
    B = torch.randn(M, m, N, device="cuda")
    G = B.mT @ B
    Qref, Rref = torch.linalg.qr(B, mode="reduced")
    P = (m + rpl - 1) // rpl
    print(f"\n### [{M},{m},{N}] rpl={rpl} -> P={P} ({'byes' if P & (P-1) else 'pow2'})")

    R = ext.tsqr_factor(B, rpl, 256)
    err_R = ((R.mT @ R - G).norm() / G.norm()).item()
    sgn = (torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
           * torch.sign(torch.diagonal(Rref, dim1=-2, dim2=-1)))
    print(f"    RtR-BtB = {err_R:.1e}")

    t_f = ev(lambda: ext.tsqr_factor(B, rpl, 256))
    t_g = ev(lambda: torch.geqrf(B))
    print(f"    factor : cuSOLVER {t_g:8.2f} ms | ours {t_f:7.2f} ms  -> {t_g/t_f:5.2f}x")

    for k in ks:
        v = torch.randn(M, N, k, device="cuda")
        ext.tsqr_factor(B, rpl, 256)
        Qv = ext.tsqr_applyQ(v, 256)
        err_q = ((Qv - (Qref * sgn.unsqueeze(1)) @ v).norm() / (Qref @ v).norm()).item()
        vpad = torch.zeros(M, m, k, device="cuda"); vpad[:, :N] = v

        def cus():
            a, tau = torch.geqrf(B)
            return torch.ormqr(a, tau, vpad, left=True, transpose=False)

        def ours():
            ext.tsqr_factor(B, rpl, 256)
            return ext.tsqr_applyQ(v, 256)

        t1, t2 = ev(cus), ev(ours)
        print(f"    full k={k:<4}: cuSOLVER {t1:8.2f} | ours {t2:7.2f}  -> {t1/t2:5.2f}x"
              f"   (Q@v err {err_q:.1e})")
        del v, vpad, Qv
        torch.cuda.empty_cache()
    if N == 232 and m == 8192:   # TensorCore (TF32) on the WY GEMM apply
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in (105, 232):
            v = torch.randn(M, N, k, device="cuda")
            ext.tsqr_factor(B, rpl, 256)
            Qv = ext.tsqr_applyQ(v, 256)
            err_q = ((Qv - (Qref * sgn.unsqueeze(1)) @ v).norm() / (Qref @ v).norm()).item()
            t2 = ev(lambda: (ext.tsqr_factor(B, rpl, 256), ext.tsqr_applyQ(v, 256)))
            print(f"    full k={k:<4} [TF32]: ours {t2:7.2f} ms   (Q@v err {err_q:.1e})")
            del v, Qv; torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
    del B, G, Qref, Rref, R
    torch.cuda.empty_cache()
