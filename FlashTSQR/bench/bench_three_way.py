"""Honest three-way comparison: torch.linalg.qr vs. Newton-Schulz (Appendix-F
exact/Cholesky-corrected variant) vs. FlashTSQR, all running the *identical*
end-to-end FraQ backend pipeline including the final B_g = Q @ eigvecs
reconstruction (the tall-skinny apply step FlashTSQR's applyQ specifically
targets). This fixes bench_vs_ns.py / diag_ns_orth_qrfree.py's mismatched
comparison, which measured NS without that final apply step.

Pipeline (keep-all rank k=N, same convention as bench_vs_ns.py):
  QR       : Q,R  = qr(B);            H = R@A;                Bg = Q  @ vecs
  NS-exact : Qt    = NS(B, steps);     L = chol(QtᵀQt);         Qe = Qt @ L^-T
             H = Lᵀ (QtᵀQt)^-1 QtᵀB @ A;                       Bg = Qe @ vecs
  FlashTSQR: R = tsqr_factor(B);       H = R@A;                Bg = tsqr_applyQ(vecs)
All three: gram = H@Hᵀ; eigh(gram) -> vecs (descending, top-N kept).
"""
import pathlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline

K = pathlib.Path(__file__).resolve().parent.parent / "kernels"
CPP = ("torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);\n"
       "torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb);")
print("compiling ...", flush=True)
ext = load_inline(name="tsqr_three_way", cpp_sources=CPP,
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


def newton_schulz5(G, steps, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transpose = X.shape[-2] > X.shape[-1]
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        Ai = X @ X.mT
        Bi = b * Ai + c * Ai @ Ai
        X = a * X + Bi @ X
    if transpose:
        X = X.mT
    return X


M, m, in_dim = 36, 768, 768
NS_STEPS = 10
CASES = [(40, 128), (80, 128), (160, 192), (232, 232)]

print(f"NS steps = {NS_STEPS}\n")
print(f"{'N':>5}{'QR (ms)':>10}{'NS-exact (ms)':>16}{'FlashTSQR (ms)':>17}"
      f"{'NS speedup':>12}{'FTSQR speedup':>15}")

def rel_error(b_ref, a_ref, b_test, a_test):
    """Basis/sign-invariant relative Frobenius distance between two rank-p
    reconstructions B@A (batched), via inner products in the small core space
    -- eigh's per-column sign/rotation ambiguity cancels out here because both
    B and its correctly-paired A carry the same (possibly flipped) sign."""
    def fro_sq(bx, ax, by, ay):
        t1 = (bx.mT @ bx) @ (ax @ ax.mT)
        t2 = (by.mT @ by) @ (ay @ ay.mT)
        t3 = (bx.mT @ by) @ (ay @ ax.mT)
        return t1.diagonal(dim1=-2, dim2=-1).sum(-1) + t2.diagonal(dim1=-2, dim2=-1).sum(-1) \
            - 2 * t3.diagonal(dim1=-2, dim2=-1).sum(-1)

    diff_sq = fro_sq(b_ref, a_ref, b_test, a_test).clamp_min(0)
    ref_sq = (b_ref.mT @ b_ref @ (a_ref @ a_ref.mT)).diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)
    return (diff_sq / ref_sq).sqrt().mean().item()


for N, rpl in CASES:
    torch.manual_seed(0)
    B = torch.randn(M, m, N, device="cuda") * 0.1
    A = torch.randn(M, N, in_dim, device="cuda") * 0.1
    k = N
    eye_N = torch.eye(N, device="cuda").unsqueeze(0)

    def qr_backend():
        Qb, Rb = torch.linalg.qr(B, mode="reduced")
        Hb = Rb @ A
        Gb = Hb @ Hb.mT
        valsb, vecsb = torch.linalg.eigh(Gb)
        valsb, vecsb = valsb.flip(-1)[:, :k], vecsb.flip(-1)[:, :, :k].contiguous()
        root = valsb.clamp_min(0).sqrt().sqrt()
        inv_root = torch.where(root > 1e-12, root.reciprocal(), torch.zeros_like(root))
        ag = (vecsb.mT @ Hb) * inv_root.unsqueeze(-1)
        bg = (Qb @ vecsb) * root.unsqueeze(-2)
        return bg, ag

    def ns_exact_backend():
        Qt = newton_schulz5(B, NS_STEPS)
        gram_qq = Qt.mT @ Qt
        L = torch.linalg.cholesky(gram_qq + 1e-8 * gram_qq.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * eye_N)
        c_tilde = torch.linalg.solve(gram_qq, Qt.mT @ B)
        Qe = torch.linalg.solve_triangular(L, Qt.mT, upper=False).mT
        He = L.mT @ c_tilde @ A
        Ge = He @ He.mT
        valse, vecse = torch.linalg.eigh(Ge)
        valse, vecse = valse.flip(-1)[:, :k], vecse.flip(-1)[:, :, :k].contiguous()
        root = valse.clamp_min(0).sqrt().sqrt()
        inv_root = torch.where(root > 1e-12, root.reciprocal(), torch.zeros_like(root))
        ag = (vecse.mT @ He) * inv_root.unsqueeze(-1)
        bg = (Qe @ vecse) * root.unsqueeze(-2)
        return bg, ag

    def flashtsqr_backend():
        Rf = ext.tsqr_factor(B, rpl, 256)
        Hf = Rf @ A
        Gf = Hf @ Hf.mT
        valsf, vecsf = torch.linalg.eigh(Gf)
        valsf, vecsf = valsf.flip(-1)[:, :k], vecsf.flip(-1)[:, :, :k].contiguous()
        root = valsf.clamp_min(0).sqrt().sqrt()
        inv_root = torch.where(root > 1e-12, root.reciprocal(), torch.zeros_like(root))
        ag = (vecsf.mT @ Hf) * inv_root.unsqueeze(-1)
        bg = ext.tsqr_applyQ(vecsf, 256) * root.unsqueeze(-2)
        return bg, ag

    # correctness: compare the actual rank-p reconstruction B_g@A_g, sign/basis-invariant
    bg_qr, ag_qr = qr_backend()
    bg_ns, ag_ns = ns_exact_backend()
    bg_ft, ag_ft = flashtsqr_backend()
    err_ns = rel_error(bg_qr, ag_qr, bg_ns, ag_ns)
    err_ft = rel_error(bg_qr, ag_qr, bg_ft, ag_ft)

    t_qr, _ = ev_time(lambda: qr_backend())
    t_ns, _ = ev_time(lambda: ns_exact_backend())
    t_ft, _ = ev_time(lambda: flashtsqr_backend())

    print(f"{N:>5}{t_qr:>10.3f}{t_ns:>16.3f}{t_ft:>17.3f}"
          f"{t_qr / t_ns:>11.2f}x{t_qr / t_ft:>14.2f}x"
          f"   [rel err vs QR: NS={err_ns:.1e} FlashTSQR={err_ft:.1e}]")

    del B, A, bg_qr, ag_qr, bg_ns, ag_ns, bg_ft, ag_ft
    torch.cuda.empty_cache()
