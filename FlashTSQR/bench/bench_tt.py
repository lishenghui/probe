"""TT-structured merge (triangle-on-triangle) vs the dense-merge kernel vs cuSOLVER.

The merge tree dominates at large N (at [28,4096,128] it is ~62% of the factor
flops), and its input is two stacked upper triangles -- the TT kernel touches
only the j+2 active rows per Householder column, cutting merge flops ~10x.

Correctness: R^T R == B^T B, and Q@v matches torch (sign-aligned), for BOTH kernels.
"""
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline

K = pathlib.Path(__file__).resolve().parent.parent / "kernels"
CPP = ("torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);\n"
       "torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb);")

print("compiling base + tt ...", flush=True)
base = load_inline(name="tsqr_base_cmp", cpp_sources=CPP,
                   cuda_sources=(K / "tsqr_dense_merge.cu").read_text(),
                   functions=["tsqr_factor", "tsqr_applyQ"], verbose=False,
                   extra_cuda_cflags=["-O3"])
tt = load_inline(name="tsqr_tt_cmp", cpp_sources=CPP,
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

for (M, m, N, rpl) in [(56, 8192, 64, 128), (56, 3072, 64, 96), (28, 4096, 128, 128)]:
    torch.manual_seed(0)
    B = torch.randn(M, m, N, device="cuda")
    G = B.mT @ B
    Qref, Rref = torch.linalg.qr(B, mode="reduced")
    print(f"\n### [{M},{m},{N}] rpl={rpl}")

    # ---- correctness of the TT kernel ----
    R_tt = tt.tsqr_factor(B, rpl, 256)
    err_R = ((R_tt.mT @ R_tt - G).norm() / G.norm()).item()
    sgn = (torch.sign(torch.diagonal(R_tt, dim1=-2, dim2=-1))
           * torch.sign(torch.diagonal(Rref, dim1=-2, dim2=-1)))
    for k in (8, N):
        v = torch.randn(M, N, k, device="cuda")
        tt.tsqr_factor(B, rpl, 256)
        Qv = tt.tsqr_applyQ(v, 256)
        err_q = ((Qv - (Qref * sgn.unsqueeze(1)) @ v).norm() / (Qref @ v).norm()).item()
        print(f"    correctness  RtR-BtB={err_R:.1e}   Q@v(k={k})={err_q:.1e}")
        del v, Qv

    # ---- factor-only timing ----
    t_cus = ev(lambda: torch.geqrf(B))
    t_b = ev(lambda: base.tsqr_factor(B, rpl, 256))
    t_t = ev(lambda: tt.tsqr_factor(B, rpl, 256))
    print(f"    factor only : cuSOLVER {t_cus:8.2f} | dense {t_b:7.2f} | TT {t_t:7.2f}"
          f"  -> TT vs dense {t_b/t_t:5.2f}x, TT vs cuSOLVER {t_cus/t_t:5.2f}x")

    # ---- full operator (factor + apply) at k=8 and k=N ----
    for k in (8, N):
        v = torch.randn(M, N, k, device="cuda")
        vpad = torch.zeros(M, m, k, device="cuda"); vpad[:, :N] = v

        def cus():
            a, tau = torch.geqrf(B)
            return torch.ormqr(a, tau, vpad, left=True, transpose=False)

        def f_base():
            base.tsqr_factor(B, rpl, 256)
            return base.tsqr_applyQ(v, 256)

        def f_tt():
            tt.tsqr_factor(B, rpl, 256)
            return tt.tsqr_applyQ(v, 256)

        t1, t2, t3 = ev(cus), ev(f_base), ev(f_tt)
        print(f"    full k={k:<4}: cuSOLVER {t1:8.2f} | dense {t2:7.2f} | TT {t3:7.2f}"
              f"  -> TT vs dense {t2/t3:5.2f}x, TT vs cuSOLVER {t1/t3:5.2f}x")
        del v, vpad
        torch.cuda.empty_cache()
    del B, G, Qref, Rref, R_tt
    torch.cuda.empty_cache()
