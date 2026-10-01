"""LoRAForge feasibility study, part 3: is fused-adapter compression CONCEPT-FAIR?

Compressing a fused multi-concept adapter is not a plain low-rank problem. The
concepts in a composition have wildly different magnitudes and ranks -- in the
ComposLoRA set a character LoRA is rank 128 and a style LoRA rank 32, and their
Delta W norms differ by an order of magnitude. A rank-k truncation that
minimises the *total* Frobenius error will happily spend its whole budget on
the loudest concept and delete the quiet one.

This script measures that on REAL multi-concept LoRAs (ComposLoRA: character /
clothing / style / background / object, SD1.5), and compares rank-allocation
policies at an identical total budget k:

  global    top-k SVD of the fused Delta W. Frobenius-optimal, concept-blind.
  equal     k/n per concept.
  prop-r    k_i proportional to the concept's own trained rank.
  prop-e    k_i proportional to ||Delta W_i||_F^2  (energy -- the greedy policy).
  balanced  water-fill k_i so every concept retains the SAME fraction of its own
            spectral energy. The policy LoRAForge would use.

Reported per policy: total relative error (what global SVD optimises) and the
per-concept subspace-preservation error (what the user actually sees), in
particular its WORST case across concepts.
"""

import argparse
import collections
import glob
import json
import os
import re
import statistics

import torch
from safetensors.torch import load_file

COMPOS = "/nobackup/proj/disk/bloom/personal/shenghui/Multi-LoRA-Composition/models/lora"


# ------------------------------------------------------------------- loading

def load_lora(path, device, prefix="lora_unet"):
    """-> {module_name: (A [r,din], B [dout,r], scale)}"""
    sd = load_file(path)
    mods = {}
    for k in sd:
        if not k.endswith(".lora_down.weight"):
            continue
        name = k[: -len(".lora_down.weight")]
        if prefix and not name.startswith(prefix):
            continue
        A = sd[name + ".lora_down.weight"]
        B = sd[name + ".lora_up.weight"]
        if A.ndim == 4:                      # conv LoRA -> flatten to 2-D
            A = A.flatten(1)
            B = B.flatten(1)
        r = A.shape[0]
        alpha = sd.get(name + ".alpha", torch.tensor(float(r))).item()
        mods[name] = (A.float().to(device), B.float().to(device), alpha / r)
    return mods


# ------------------------------------------------------- allocation policies

def waterfill_balanced(spectra, k):
    """Pick k_i so each concept keeps the same fraction of its own energy."""
    n = len(spectra)
    cum = [torch.cumsum(s ** 2, 0) / (s ** 2).sum().clamp_min(1e-30) for s in spectra]
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        ks = [int(torch.searchsorted(c, mid).item()) + 1 for c in cum]
        ks = [min(k_i, len(s)) for k_i, s in zip(ks, spectra)]
        if sum(ks) > k:
            hi = mid
        else:
            lo = mid
    ks = [int(torch.searchsorted(c, lo).item()) + 1 for c in cum]
    ks = [max(1, min(k_i, len(s))) for k_i, s in zip(ks, spectra)]
    # spend any leftover budget on the concept with the most energy still lost
    while sum(ks) < k:
        best, bi = -1.0, -1
        for i, (s, k_i) in enumerate(zip(spectra, ks)):
            if k_i >= len(s):
                continue
            lost = (s[k_i:] ** 2).sum().item()
            if lost > best:
                best, bi = lost, i
        if bi < 0:
            break
        ks[bi] += 1
    while sum(ks) > k:                       # trim if we overshot
        i = max(range(n), key=lambda i: ks[i])
        ks[i] -= 1
    return ks


def allocate(policy, spectra, ranks, k):
    n = len(spectra)
    if policy == "equal":
        ks = [k // n] * n
    elif policy == "prop-r":
        tot = sum(ranks)
        ks = [max(1, round(k * r / tot)) for r in ranks]
    elif policy == "prop-e":
        e = [float((s ** 2).sum()) for s in spectra]
        tot = sum(e)
        ks = [max(1, round(k * x / tot)) for x in e]
    elif policy == "balanced":
        return waterfill_balanced(spectra, k)
    else:
        raise ValueError(policy)
    ks = [min(k_i, len(s)) for k_i, s in zip(ks, spectra)]
    while sum(ks) > k:
        i = max(range(n), key=lambda i: ks[i])
        ks[i] -= 1
    return ks


# ------------------------------------------------------------------ metrics

def concept_errors(dWs, U_keep, V_keep):
    """||dW_i - P_U dW_i P_V||_F / ||dW_i||_F  -- how much of concept i survives."""
    errs = []
    for dW in dWs:
        proj = U_keep @ (U_keep.T @ dW @ V_keep) @ V_keep.T
        errs.append(float((dW - proj).norm() / dW.norm().clamp_min(1e-30)))
    return errs


def keep_subspaces(factors):
    """factors: list of (Bk [dout,ki], Ak [ki,din]) -> orthobases of the union."""
    B = torch.cat([b for b, _ in factors], dim=1)
    A = torch.cat([a for _, a in factors], dim=0)
    U, _ = torch.linalg.qr(B, mode="reduced")
    V, _ = torch.linalg.qr(A.T, mode="reduced")
    return U, V, B, A


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="reality", choices=["reality", "anime"])
    ap.add_argument("--concepts", nargs="+",
                    default=["character_1", "clothing_1", "style_1"],
                    help="the composition (mimics content+style+motion)")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="per-concept merge weights (default 0.8 each, as ComposLoRA)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    ap.add_argument("--max-modules", type=int, default=64, help="0 = all")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "")

    n = len(args.concepts)
    w = args.weights or [0.8] * n
    assert len(w) == n

    loras = []
    for c in args.concepts:
        p = os.path.join(COMPOS, args.domain, c + ".safetensors")
        m = load_lora(p, dev)
        r = statistics.median([A.shape[0] for A, _, _ in m.values()])
        print(f"  {c:<14} {len(m):>4} unet modules, median rank {int(r)}, "
              f"{os.path.getsize(p) / 2**20:.0f} MiB")
        loras.append(m)

    shared = sorted(set.intersection(*[set(m) for m in loras]))
    if args.max_modules:
        shared = shared[:: max(1, len(shared) // args.max_modules)][: args.max_modules]
    print(f"\n{len(shared)} shared modules used "
          f"(of {len(set.intersection(*[set(m) for m in loras]))})\n")

    # ---------------------------------------------- per-concept magnitude gap
    norms = collections.defaultdict(list)
    for name in shared:
        for c, m, wi in zip(args.concepts, loras, w):
            A, B, s = m[name]
            norms[c].append(float((wi * s * (B @ A)).norm()))
    print("### magnitude imbalance across concepts (||w_i * Delta W_i||_F)")
    print(f"{'concept':<14} | {'median':>9} | {'min':>9} | {'max':>9} | {'rank':>5}")
    med = {}
    for c, m in zip(args.concepts, loras):
        v = norms[c]
        med[c] = statistics.median(v)
        r = int(statistics.median([A.shape[0] for A, _, _ in m.values()]))
        print(f"{c:<14} | {med[c]:9.3f} | {min(v):9.3f} | {max(v):9.3f} | {r:5d}")
    loud, quiet = max(med, key=med.get), min(med, key=med.get)
    print(f"-> loudest / quietest = {med[loud] / max(med[quiet], 1e-9):.1f}x "
          f"({loud} vs {quiet})")

    # -------------------------------------------------------- policy compare
    policies = ["global", "equal", "prop-r", "prop-e", "balanced"]
    out = {"concepts": args.concepts, "domain": args.domain, "budgets": {}}

    for k in args.budgets:
        acc = {p: {"tot": [], "per": collections.defaultdict(list), "ks": None}
               for p in policies}
        for name in shared:
            dWs, spectra, ranks, svds = [], [], [], []
            for m, wi in zip(loras, w):
                A, B, s = m[name]
                dW = wi * s * (B @ A)
                dWs.append(dW)
                U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
                svds.append((U, S, Vh))
                spectra.append(S)
                ranks.append(A.shape[0])
            dW_sum = sum(dWs)
            nrm = dW_sum.norm().clamp_min(1e-30)
            kk = min(k, min(dW_sum.shape))

            # --- global: Frobenius-optimal top-k of the fused matrix
            U, S, Vh = torch.linalg.svd(dW_sum, full_matrices=False)
            Uk, Vk = U[:, :kk].contiguous(), Vh[:kk].T.contiguous()
            approx = Uk @ (Uk.T @ dW_sum @ Vk) @ Vk.T
            acc["global"]["tot"].append(float((dW_sum - approx).norm() / nrm))
            for c, e in zip(args.concepts, concept_errors(dWs, Uk, Vk)):
                acc["global"]["per"][c].append(e)
            acc["global"]["ks"] = [kk]

            # --- per-concept allocations
            for p in policies[1:]:
                ks = allocate(p, spectra, ranks, kk)
                factors = []
                for (U_, S_, Vh_), ki in zip(svds, ks):
                    ki = max(1, min(ki, S_.numel()))
                    factors.append((U_[:, :ki] * S_[:ki], Vh_[:ki]))
                Uu, Vv, Bc, Ac = keep_subspaces(factors)
                approx = Bc @ Ac
                acc[p]["tot"].append(float((dW_sum - approx).norm() / nrm))
                for c, e in zip(args.concepts, concept_errors(dWs, Uu, Vv)):
                    acc[p]["per"][c].append(e)
                acc[p]["ks"] = ks

        print(f"\n### budget k = {k}   (per-concept error = fraction of that "
              f"concept's operator lost)")
        hdr = (f"{'policy':<9} | {'k_i':<14} | {'total err':>9} | "
               + " | ".join(f"{c[:11]:>11}" for c in args.concepts) + f" | {'worst':>7}")
        print(hdr)
        print("-" * len(hdr))
        for p in policies:
            tot = statistics.median(acc[p]["tot"])
            per = [statistics.median(acc[p]["per"][c]) for c in args.concepts]
            ks = ",".join(map(str, acc[p]["ks"]))
            print(f"{p:<9} | {ks:<14} | {tot:9.4f} | "
                  + " | ".join(f"{e:11.4f}" for e in per)
                  + f" | {max(per):7.4f}")
        out["budgets"][k] = {p: {"total": statistics.median(acc[p]["tot"]),
                                 "per": {c: statistics.median(acc[p]["per"][c])
                                         for c in args.concepts},
                                 "ks": acc[p]["ks"]} for p in policies}

    print("\n### reading of the table")
    print("`global` minimises TOTAL error by construction. If its `worst` column is")
    print("much larger than `balanced`'s, a concept-blind rank-k fusion is silently")
    print("deleting a concept -- and that is the algorithmic opening for LoRAForge.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
