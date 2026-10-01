#!/usr/bin/env python3
"""Fixed-budget spectral compression using the FlashTSQR kernel on GPU.

Same allocation and same truncation as compress_oft.py; only the tall-skinny QR
moves. 439 modules x 2 factors is 878 QRs per adapter, and on the MLP shapes
(11008 x 32) torch's CPU path took about twenty minutes for four adapters.

The kernel is stateful: tsqr_factor stores the Householder vectors that the
following tsqr_applyQ consumes. The core SVD needs both R factors before either
Q can be applied, so A is factored twice -- once for its R, once to restore its
state for applyQ. Three factors and two applies per module, still far below the
cost of forming Q explicitly.
"""
from __future__ import annotations

import argparse, json, pathlib, shutil, time
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.cpp_extension import load_inline

CPP = ("torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb);\n"
       "torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb);")
ROOT = pathlib.Path("/nobackup/proj/disk/bloom/personal/shenghui/probe/FlashTSQR")


def build():
    t0 = time.perf_counter()
    ext = load_inline(name="tsqr_full", cpp_sources=CPP,
                      cuda_sources=(ROOT / "kernels" / "tsqr_full.cu").read_text(),
                      functions=["tsqr_factor", "tsqr_applyQ"], verbose=False,
                      extra_cuda_cflags=["-O3"])
    print(f"kernel compiled in {time.perf_counter()-t0:.0f}s", flush=True)
    return ext


def truncate(ext, A, B, keep, rpl=256, tpb=256):
    """Rank-`keep` factors of dW = B@A, zero-padded back to the nominal rank."""
    r = A.shape[0]
    A2 = (A.reshape(r, -1) if A.ndim > 2 else A).float().cuda()
    B2 = (B.reshape(B.shape[0], B.shape[1]) if B.ndim > 2 else B).float().cuda()
    At = A2.T.contiguous()

    Ra = ext.tsqr_factor(At.unsqueeze(0), rpl, tpb)[0]
    Rb = ext.tsqr_factor(B2.unsqueeze(0), rpl, tpb)[0]  # B's state is live from here
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)

    Bn = torch.zeros(B2.shape[0], r, device="cuda")
    An = torch.zeros(r, At.shape[0], device="cuda")
    if keep:
        Bn[:, :keep] = ext.tsqr_applyQ((U[:, :keep] * S[:keep]).unsqueeze(0).contiguous(), tpb)[0]
        ext.tsqr_factor(At.unsqueeze(0), rpl, tpb)  # restore A's state for its applyQ
        An[:keep] = ext.tsqr_applyQ(Vh[:keep].T.unsqueeze(0).contiguous(), tpb)[0].T
    return An, Bn, S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--budget-frac", type=float, default=0.70)
    ap.add_argument("--adapter-only", action="store_true")
    ap.add_argument("--reference", type=pathlib.Path, default=None,
                    help="a CPU-produced adapter to check this one against")
    args = ap.parse_args()
    if not 0 < args.budget_frac <= 1:
        ap.error("--budget-frac must be in (0, 1]")
    if args.output.exists():
        ap.error("output already exists; choose a new directory")

    ext = build()
    print("compression backend: FlashTSQR/kernels/tsqr_full.cu", flush=True)
    src = args.source / "lora_adapter"
    w = load_file(src / "adapter_model.safetensors")
    keys = sorted(k for k in w if ".lora_A" in k)

    t0 = time.perf_counter()
    spec = {}
    for k in keys:
        A = w[k]; B = w[k.replace(".lora_A", ".lora_B")]
        _, _, S = truncate(ext, A, B, 0)
        spec[k] = S.cpu()
    pool = sorted(((float(s), k, i) for k in keys for i, s in enumerate(spec[k])), key=lambda t: -t[0])
    budget = int(round(args.budget_frac * len(pool)))
    kept = {}
    for _, k, _ in pool[:budget]:
        kept[k] = kept.get(k, 0) + 1

    out = dict(w); ranks = {}; ek = ea = 0.0
    for k in keys:
        A = w[k]; B = w[k.replace(".lora_A", ".lora_B")]
        keep = kept.get(k, 0); ranks[k] = keep
        S = spec[k]; ea += float((S**2).sum()); ek += float((S[:keep]**2).sum())
        An, Bn, _ = truncate(ext, A, B, keep)
        out[k] = An.reshape(A.shape).to(w[k].dtype).cpu()
        out[k.replace(".lora_A", ".lora_B")] = Bn.reshape(B.shape).to(B.dtype).cpu()
        if not torch.isfinite(out[k]).all() or not torch.isfinite(out[k.replace(".lora_A", ".lora_B")]).all():
            raise RuntimeError(f"Non-finite compressed factors: {k}")
    dt = time.perf_counter() - t0

    dst = args.output / "lora_adapter"; dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    for item in ([] if args.adapter_only else args.source.iterdir()):
        if item.name != "lora_adapter":
            (shutil.copytree if item.is_dir() else shutil.copy)(item, args.output / item.name)
    nz = [v for v in ranks.values() if v > 0]
    original_params = sum(w[k].numel() + w[k.replace(".lora_A", ".lora_B")].numel() for k in keys)
    compact_params = sum(ranks[k] * (w[k].numel() // w[k].shape[0] +
                        w[k.replace(".lora_A", ".lora_B")].numel() // w[k].shape[0]) for k in keys)
    manifest = dict(backend="flash_tsqr", kernel=str(ROOT / "kernels" / "tsqr_full.cu"),
                    source=str(args.source), output=str(args.output), budget_frac=args.budget_frac,
                    directions_total=len(pool), directions_kept=budget, energy_retained=ek/ea,
                    modules=len(keys), modules_at_zero=len(keys)-len(nz),
                    rank_min=min(nz, default=0), rank_max=max(nz, default=0),
                    rank_mean=float(np.mean(list(ranks.values()))),
                    original_ab_parameters=original_params, compact_ab_parameters=compact_params,
                    zero_padded=True, compression_s=dt,
                    ranks={k.replace("base_model.model.", ""): v for k, v in ranks.items()})
    (args.output / "compression.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.source.name}: {budget}/{len(pool)} directions in {dt:.0f}s, "
          f"energy {ek/ea:.4f}, ranks {min(nz, default=0)}-{max(nz, default=0)} mean {np.mean(list(ranks.values())):.1f}, "
          f"{len(keys)-len(nz)} at zero", flush=True)

    if args.reference:
        ref = load_file(args.reference / "lora_adapter" / "adapter_model.safetensors")
        # dW is the invariant; the factors themselves are only unique up to an
        # orthogonal rotation, so compare the products, not the tensors.
        worst = 0.0
        for k in keys[:40]:
            bk = k.replace(".lora_A", ".lora_B")
            g = (out[bk].float().reshape(out[bk].shape[0], -1) @ out[k].float().reshape(out[k].shape[0], -1))
            h = (ref[bk].float().reshape(ref[bk].shape[0], -1) @ ref[k].float().reshape(ref[k].shape[0], -1))
            worst = max(worst, float((g - h).norm() / h.norm().clamp_min(1e-12)))
        print(f"vs CPU reference: worst relative dW error over 40 modules = {worst:.3e}", flush=True)


if __name__ == "__main__":
    main()
