#!/usr/bin/env python3
"""Compress an OpenVLA-OFT adapter to a fixed budget of retained directions.

Allocation is global spectral water-filling: pool every module's squared
singular values, keep the largest `budget` of them, and each module's rank is
however many of its own values survived. For a Frobenius objective at a fixed
total rank this is optimal, and it is the spectral rung the functional method
has to beat -- not a stand-in for it.

Truncation never forms dW. With B = Qb Rb and A^T = Qa Ra, dW = Qb (Rb Ra^T) Qa^T,
so an r x r SVD of the core gives the exact rank-k factors.

The output keeps rank 32 and zero-pads the dropped directions. That is
numerically identical to compact factors (verified separately) and loads as an
ordinary adapter; it costs storage and compute that a compact export would save,
which is irrelevant to the question this answers -- how much task success a
budget costs.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def factors(A: torch.Tensor, B: torch.Tensor):
    """(Qb, core, Qa) with dW = Qb @ core @ Qa^T, core r x r."""
    A2 = A.reshape(A.shape[0], -1) if A.ndim > 2 else A
    B2 = B.reshape(B.shape[0], B.shape[1]) if B.ndim > 2 else B
    Qb, Rb = torch.linalg.qr(B2.float(), mode="reduced")
    Qa, Ra = torch.linalg.qr(A2.float().T, mode="reduced")
    return Qb, Rb @ Ra.T, Qa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--budget-frac", type=float, default=0.70)
    ap.add_argument("--adapter-only", action="store_true",
                    help="write adapter and manifest only; use shared --metadata for evaluation")
    args = ap.parse_args()
    if not 0 < args.budget_frac <= 1:
        ap.error("--budget-frac must be in (0, 1]")
    if args.output.exists():
        ap.error("output already exists; choose a new directory")

    src_adapter = args.source / "lora_adapter"
    w = load_file(src_adapter / "adapter_model.safetensors")
    keys = sorted(k for k in w if ".lora_A" in k)

    cache, pool = {}, []
    for k in keys:
        B = w[k.replace(".lora_A", ".lora_B")]
        Qb, core, Qa = factors(w[k], B)
        U, S, Vh = torch.linalg.svd(core, full_matrices=False)
        cache[k] = (Qb, U, S, Vh, Qa, w[k].shape, B.shape)
        pool.extend((float(s), k, i) for i, s in enumerate(S))

    total = len(pool)
    budget = int(round(args.budget_frac * total))
    pool.sort(key=lambda t: -t[0])
    kept = {}
    for _, k, _ in pool[:budget]:
        kept[k] = kept.get(k, 0) + 1

    out = dict(w)
    energy_kept = energy_all = 0.0
    ranks = {}
    for k in keys:
        Qb, U, S, Vh, Qa, shA, shB = cache[k]
        r = w[k].shape[0]
        keep = kept.get(k, 0)
        energy_all += float((S**2).sum())
        energy_kept += float((S[:keep]**2).sum())
        ranks[k] = keep
        # B' = Qb U_k S_k, A' = V_k^T Qa^T, zero-padded back to rank r
        Bn = torch.zeros(Qb.shape[0], r)
        An = torch.zeros(r, Qa.shape[0])
        if keep:
            Bn[:, :keep] = Qb @ (U[:, :keep] * S[:keep])
            An[:keep] = Vh[:keep] @ Qa.T
        out[k] = An.reshape(shA).to(w[k].dtype)
        out[k.replace(".lora_A", ".lora_B")] = Bn.reshape(shB).to(w[k].dtype)

    dst = args.output / "lora_adapter"
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "adapter_model.safetensors"))
    shutil.copy(src_adapter / "adapter_config.json", dst / "adapter_config.json")
    for item in ([] if args.adapter_only else args.source.iterdir()):
        if item.name != "lora_adapter":
            (shutil.copytree if item.is_dir() else shutil.copy)(item, args.output / item.name)

    nz = [v for v in ranks.values() if v > 0]
    manifest = dict(source=str(args.source), budget_frac=args.budget_frac,
                    directions_total=total, directions_kept=budget,
                    energy_retained=energy_kept / energy_all,
                    modules=len(keys), modules_at_zero=len(keys) - len(nz),
                    rank_min=int(min(nz)), rank_max=int(max(nz)),
                    rank_mean=float(np.mean(list(ranks.values()))),
                    ranks={k.replace("base_model.model.", ""): v for k, v in ranks.items()})
    (args.output / "compression.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.source.name}: {budget}/{total} directions ({args.budget_frac:.0%}), "
          f"energy retained {energy_kept/energy_all:.4f}, ranks {min(nz)}-{max(nz)} "
          f"(mean {np.mean(list(ranks.values())):.1f}), {len(keys)-len(nz)} modules at zero", flush=True)


if __name__ == "__main__":
    main()
