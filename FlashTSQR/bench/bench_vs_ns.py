"""FlashTSQR as the FraQ orthogonalization backend, benchmarked on the exact
same synthetic shapes used in diag_ns_orth_qrfree.py (36 modules, out=in=768,
rank_sum in {40, 80, 160, 232, 320}) so the numbers are directly comparable to
the QR-vs-Newton-Schulz table in the top-level README.

Pipeline per rank_sum N (mirrors aggregate.py's _fraq_rank / the paper's core
decomposition, but using FlashTSQR's R factor + Q@v instead of torch.linalg.qr):
    R      = tsqr_factor(B, rows_per_leaf, tpb)      # [M, N, N] triangular
    H      = R @ A                                    # [M, N, in]  cheap GEMM
    G      = H @ H^T ; eigh(G) -> vecs [M, N, p]       # core-space spectrum
    B_g    = tsqr_applyQ(vecs[:, :, :p], tpb)          # [M, out, p]  <- FlashTSQR's job
Correctness check: R^T R == B^T B (same identity used throughout FlashTSQR's
README), and the recovered top singular values are compared against
torch.linalg.qr's reference.
"""
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline

K = pathlib.Path(__file__).resolve().parent.parent / "kernels"
CPP = ("torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);\n"
       "torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb);")
print("compiling ...", flush=True)
ext = load_inline(name="tsqr_vs_ns", cpp_sources=CPP,
                  cuda_sources=(K / "tsqr_full.cu").read_text(),
                  functions=["tsqr_factor", "tsqr_applyQ"], verbose=False,
                  extra_cuda_cflags=["-O3"])
print("compiled.\n", flush=True)

print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cuda.matmul.allow_tf32 = False


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


M, m, in_dim = 36, 768, 768
CASES = [(40, 128), (80, 128), (160, 192), (232, 232), (320, 320)]

print(f"{'N':>5}{'rpl':>6}{'QR+eigh (ms)':>15}{'FlashTSQR+eigh (ms)':>21}{'speedup':>10}"
      f"{'R err':>12}{'sigma err':>12}")

for N, rpl in CASES:
    torch.manual_seed(0)
    B = torch.randn(M, m, N, device="cuda") * 0.1
    A = torch.randn(M, N, in_dim, device="cuda") * 0.1
    k = N  # keep-all: the conservative case (worst-case Q@v width)

    try:
        # ---- correctness ----
        R = ext.tsqr_factor(B, rpl, 256)
        G_ref = B.mT @ B
        err_R = ((R.mT @ R - G_ref).norm() / G_ref.norm()).item()

        Qref, Rref = torch.linalg.qr(B, mode="reduced")
        Href = Rref @ A
        gram_ref = Href @ Href.mT
        vals_ref = torch.linalg.eigh(gram_ref)[0].flip(-1).clamp_min(0).sqrt()

        H = R @ A
        gram = H @ H.mT
        vals, vecs = torch.linalg.eigh(gram)
        vals, vecs = vals.flip(-1), vecs.flip(-1)
        sigma = vals.clamp_min(0).sqrt()
        err_sigma = ((sigma - vals_ref).norm() / vals_ref.norm().clamp_min(1e-30)).item()

        vecs_k = vecs[:, :, :k].contiguous()
        Bg = ext.tsqr_applyQ(vecs_k, 256)   # [M, out, k] -- the actual FlashTSQR payoff

        # ---- timing: full backend (factor + H + gram/eigh + applyQ) vs torch QR + eigh ----
        def qr_backend():
            Qb, Rb = torch.linalg.qr(B, mode="reduced")
            Hb = Rb @ A
            Gb = Hb @ Hb.mT
            valsb, vecsb = torch.linalg.eigh(Gb)
            return Qb @ vecsb[:, :, :k].contiguous()

        def flashtsqr_backend():
            Rf = ext.tsqr_factor(B, rpl, 256)
            Hf = Rf @ A
            Gf = Hf @ Hf.mT
            valsf, vecsf = torch.linalg.eigh(Gf)
            vecsf = vecsf.flip(-1)[:, :, :k].contiguous()
            return ext.tsqr_applyQ(vecsf, 256)

        t_qr, _ = ev_time(qr_backend)
        t_fl, _ = ev_time(flashtsqr_backend)

        print(f"{N:>5}{rpl:>6}{t_qr:>15.3f}{t_fl:>21.3f}{t_qr / t_fl:>9.2f}x"
              f"{err_R:>12.2e}{err_sigma:>12.2e}")
        del B, A, R, G_ref, Qref, Rref, Href, gram_ref, H, gram, vals, vecs, vecs_k, Bg
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{N:>5}{rpl:>6}  FAILED: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
