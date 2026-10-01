"""LoRAForge feasibility study, part 2: the MERGE side (T_merge).

Given a request that names three concept LoRAs (content r32, style r32,
motion r64), how long does it take to turn them into ONE adapter that the 50
denoising steps can reuse? Measured on the real Wan 2.1 1.3B module inventory:

    240 x [1536, 1536]   attn1/attn2 q,k,v,out
     30 x [8960, 1536]   ffn.net.0.proj
     30 x [1536, 8960]   ffn.net.2
    = 300 LoRA-carrying modules, Sum r = 128

Routes timed:

  stack        concat the three (A,B) pairs -> one rank-128 adapter. Exact,
               and essentially free. The floor.
  dense        W0 += B_cat @ A_cat for all 300 modules, plus the private
               weight copy a per-request dense merge forces you to make.
  recompress   FraQ-style: QR(B_cat), QR(A_cat^T), SVD of the [128,128] core,
               truncate to rank k, re-apply Q. Three sub-routes:
                 loop      per-module torch.linalg (what a naive server does)
                 batched   grouped by shape, batched cuSOLVER
                 flashtsqr batched, our TSQR kernel (if it compiles)
  svd_dense    form Delta W [out,in] and SVD it. The strawman; shows why
               nobody does recompression online today.

Output feeds S* = T_merge / (T_3lora - T_fused) together with part 1.
"""

import argparse
import pathlib
import statistics

import torch

KERNELS = pathlib.Path(__file__).resolve().parent.parent / "kernels"

DISCO_RANKS = (32, 32, 64)
SUM_R = sum(DISCO_RANKS)

# (count, out_features, in_features) for Wan 2.1 T2V-1.3B, 30 layers
INVENTORY = [
    (30 * 8, 1536, 1536),   # attn1 q,k,v,out.0 + attn2 q,k,v,out.0
    (30 * 1, 8960, 1536),   # ffn.net.0.proj
    (30 * 1, 1536, 8960),   # ffn.net.2
]


def ev(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def make_adapters(dev, dtype):
    """Per shape group: the three concept LoRAs, as a serving system holds them."""
    groups = []
    for (M, d_out, d_in) in INVENTORY:
        As = [torch.randn(M, r, d_in, device=dev, dtype=dtype) * (1 / d_in) ** 0.5
              for r in DISCO_RANKS]
        Bs = [torch.randn(M, d_out, r, device=dev, dtype=dtype) * (1 / r) ** 0.5
              for r in DISCO_RANKS]
        W0 = torch.randn(M, d_out, d_in, device=dev, dtype=dtype) * (1 / d_in) ** 0.5
        groups.append(dict(M=M, d_out=d_out, d_in=d_in, As=As, Bs=Bs, W0=W0))
    return groups


# ------------------------------------------------------------------- routes

def r_stack(groups):
    """Exact fusion in factored form: one rank-128 adapter."""
    out = []
    for g in groups:
        out.append((torch.cat(g["As"], dim=1), torch.cat(g["Bs"], dim=2)))
    return out


def r_dense(groups):
    """W0 += sum_i B_i A_i, in place on a (private) copy of the weights."""
    for g in groups:
        A = torch.cat(g["As"], dim=1)
        B = torch.cat(g["Bs"], dim=2)
        g["W0"].baddbmm_(B, A)


def _recompress_batched(A, B, k, use_flash=None, rpl=128, tpb=256):
    """A:[M,N,din] B:[M,dout,N] -> rank-k (A2,B2) with B2@A2 ~= B@A.

    B@A = (Q_B R_B)(R_A^T Q_A^T); SVD the [N,N] core, keep k.
    """
    M, N, d_in = A.shape
    d_out = B.shape[1]
    Bf = B.float()
    Af = A.float().transpose(1, 2).contiguous()      # [M, d_in, N]

    if use_flash is not None:
        R_B = use_flash.tsqr_factor(Bf, min(rpl, d_out), tpb)
        R_A = use_flash.tsqr_factor(Af, min(rpl, d_in), tpb)
        core = R_B @ R_A.transpose(1, 2)
        U, S, Vh = torch.linalg.svd(core)
        Uk = (U[:, :, :k] * S[:, None, :k]).contiguous()
        Vk = Vh[:, :k, :].transpose(1, 2).contiguous()
        # NB: applyQ consumes the reflectors of the LAST factor call, so the
        # two applies must bracket their own factor calls; done by the caller
        # for the real pipeline. Here we re-factor to keep the measurement
        # honest (both factorisations + both applies are paid for).
        use_flash.tsqr_factor(Bf, min(rpl, d_out), tpb)
        B2 = use_flash.tsqr_applyQ(Uk, tpb)
        use_flash.tsqr_factor(Af, min(rpl, d_in), tpb)
        A2 = use_flash.tsqr_applyQ(Vk, tpb).transpose(1, 2)
        return A2, B2

    Q_B, R_B = torch.linalg.qr(Bf, mode="reduced")
    Q_A, R_A = torch.linalg.qr(Af, mode="reduced")
    core = R_B @ R_A.transpose(1, 2)
    U, S, Vh = torch.linalg.svd(core)
    B2 = Q_B @ (U[:, :, :k] * S[:, None, :k])
    A2 = (Q_A @ Vh[:, :k, :].transpose(1, 2)).transpose(1, 2)
    return A2, B2


def r_recompress(groups, k, use_flash=None):
    out = []
    for g in groups:
        A = torch.cat(g["As"], dim=1)
        B = torch.cat(g["Bs"], dim=2)
        out.append(_recompress_batched(A, B, k, use_flash))
    return out


def r_recompress_loop(groups, k):
    """No batching: one torch.linalg call per module. The naive server."""
    for g in groups:
        A = torch.cat(g["As"], dim=1).float()
        B = torch.cat(g["Bs"], dim=2).float()
        for i in range(g["M"]):
            Q_B, R_B = torch.linalg.qr(B[i], mode="reduced")
            Q_A, R_A = torch.linalg.qr(A[i].T.contiguous(), mode="reduced")
            U, S, Vh = torch.linalg.svd(R_B @ R_A.T)
            _ = Q_B @ (U[:, :k] * S[:k])
            _ = (Q_A @ Vh[:k].T).T


def r_svd_dense(groups, k, limit=32):
    """Form Delta W and SVD it. Strawman; only `limit` modules per group."""
    for g in groups:
        A = torch.cat(g["As"], dim=1).float()
        B = torch.cat(g["Bs"], dim=2).float()
        n = min(limit, g["M"])
        dW = B[:n] @ A[:n]
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        _ = U[:, :, :k] * S[:, None, :k]
        _ = Vh[:, :k, :]


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[32, 48, 64],
                    help="retained ranks for recompression")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-flash", action="store_true")
    ap.add_argument("--no-loop", action="store_true", help="skip the slow per-module route")
    args = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

    nmod = sum(M for M, _, _ in INVENTORY)
    print(f"Wan 2.1 1.3B inventory: {nmod} LoRA modules, Sum r = {SUM_R} "
          f"({'+'.join(map(str, DISCO_RANKS))})")
    for M, o, i in INVENTORY:
        print(f"    {M:>4} x [{o}, {i}]")

    flash = None
    if not args.no_flash:
        try:
            from torch.utils.cpp_extension import load_inline
            cpp = ("torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);\n"
                   "torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb);")
            flash = load_inline(name="tsqr_merge", cpp_sources=cpp,
                                cuda_sources=(KERNELS / "tsqr_full.cu").read_text(),
                                functions=["tsqr_factor", "tsqr_applyQ"],
                                verbose=False, extra_cuda_cflags=["-O3"])
            print("FlashTSQR kernel: loaded")
        except Exception as e:
            print("FlashTSQR kernel: unavailable ->", repr(e)[:200])
            flash = None

    groups = make_adapters(dev, dtype)
    bytes_w = sum(M * o * i * 2 for M, o, i in INVENTORY)
    bytes_lora = sum(M * (r * i + o * r) * 2
                     for M, o, i in INVENTORY for r in DISCO_RANKS)
    print(f"\nweights touched by a dense merge: {bytes_w / 2**30:.2f} GiB "
          f"| the three adapters: {bytes_lora / 2**20:.1f} MiB")

    print("\n### T_merge: one request's three LoRAs -> one fused adapter")
    hdr = f"{'route':>26} | {'ms':>9} | {'note':<44}"
    print(hdr)
    print("-" * len(hdr))

    res = {}

    res["stack"] = ev(lambda: r_stack(groups), reps=args.reps)
    print(f"{'stack (exact, factored)':>26} | {res['stack']:9.3f} | "
          f"{'rank-128 adapter, no quality loss':<44}")

    res["dense"] = ev(lambda: r_dense(groups), reps=args.reps)
    print(f"{'dense (W0 += BA, in place)':>26} | {res['dense']:9.3f} | "
          f"{'needs a private ' + f'{bytes_w / 2**30:.1f} GiB' + ' weight copy':<44}")

    clone_ms = ev(lambda: [g["W0"].clone() for g in groups], reps=3)
    res["weight_clone"] = clone_ms
    print(f"{'  + the weight copy':>26} | {clone_ms:9.3f} | "
          f"{'per concurrent composition':<44}")

    for k in args.k:
        t = ev(lambda k=k: r_recompress(groups, k), reps=args.reps)
        res[f"recompress_batched_k{k}"] = t
        print(f"{f'recompress k={k} (cuSOLVER)':>26} | {t:9.3f} | "
              f"{'batched QR+SVD+apply, grouped by shape':<44}")
        if flash is not None:
            tf = ev(lambda k=k: r_recompress(groups, k, use_flash=flash), reps=args.reps)
            res[f"recompress_flash_k{k}"] = tf
            print(f"{f'recompress k={k} (FlashTSQR)':>26} | {tf:9.3f} | "
                  f"{f'{t / tf:.2f}x over cuSOLVER':<44}")

    if not args.no_loop:
        k = args.k[0]
        t = ev(lambda: r_recompress_loop(groups, k), warmup=1, reps=3)
        res["recompress_loop"] = t
        print(f"{f'recompress k={k} per-module':>26} | {t:9.3f} | "
              f"{'no batching: the naive implementation':<44}")

    k = args.k[0]
    t = ev(lambda: r_svd_dense(groups, k, limit=32), warmup=1, reps=3)
    res["svd_dense_32"] = t
    scale = nmod / (32 * len(INVENTORY))
    print(f"{f'dense SVD k={k} (32/group)':>26} | {t:9.3f} | "
          f"{f'strawman; ~{t * scale:.0f} ms for all {nmod}':<44}")

    # ------------------------------------------------------------- break-even
    print("\n### break-even, given a per-step saving D = T_3lora - T_fused")
    print("S* = T_merge / D   (Wan 2.1 480p/49f runs S = 50 steps, x2 with CFG)")
    print(f"{'T_merge (ms)':>28} | " + " | ".join(f"D={d}ms" for d in (1, 2, 5, 10, 20)))
    for name in ("stack", "dense", f"recompress_batched_k{args.k[0]}",
                 f"recompress_flash_k{args.k[0]}", "recompress_loop"):
        if name not in res:
            continue
        tm = res[name]
        cells = " | ".join(f"{tm / d:6.1f}" for d in (1, 2, 5, 10, 20))
        print(f"{name + f' ({tm:.2f})':>28} | {cells}")


if __name__ == "__main__":
    main()
