#!/usr/bin/env python3
"""Allocate the rank budget by effective perturbation instead of per-layer energy.

Energy truncation fixes the *relative* residual per module: keeping 90% of the
energy leaves ||dW - dW_k|| / ||dW|| = sqrt(0.1) in every layer alike.  But what
reaches the model is the residual relative to the layer's own weight,

    P_l = || dW_l - dW_{l,k} || / || W_l ||  =  S_l * sqrt(1 - tau_l),

and S_l varies by 5x across OpenVLA's modules (p10 0.053, p90 0.211).  A uniform
tau therefore spends the budget evenly while the damage lands unevenly.

Equalising P_l instead is a global water-filling: pool every singular value from
every module as sigma^2 / ||W_l||^2, keep the largest ones until the budget runs
out, and let each module take whatever rank that implies.  Same parameter count,
different distribution.

Output matches the correction-adapter layout used for evaluation --
cat(truncated, original) and cat(truncated, -original) -- so the existing replay
harness reads it unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def base_name(a_key: str) -> str:
    name = re.sub(r"\.lora_[AB]\.(weight|default\.weight)$", ".weight", a_key)
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def as_matrix(t: torch.Tensor, rank_dim: int) -> torch.Tensor:
    """2-D view of a LoRA factor with the rank axis kept in place.

    Conv adapters store A as [r, in, kh, kw] and B as [out, r, 1, 1]; the rank
    axis is 0 for A and 1 for B, and ``reshape(-1, shape[-1])`` would silently
    fold B along its 1x1 spatial axis instead.
    """
    if rank_dim == 0:
        return t.reshape(t.shape[0], -1).float()
    return t.reshape(t.shape[0], t.shape[1]).float()


def svd_of_product(a: torch.Tensor, b: torch.Tensor):
    """Singular triplets of B @ A without forming the full matrix.

    B is [N, r] and A is [r, K] with r <= 32, so QR both sides and take the SVD
    of the r x r core.
    """
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.T)
    return qb @ u, s, vh @ qa.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True,
                        help="an existing correction adapter; its tail is the original update")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--budget-from", type=Path, required=True,
                        help="correction adapter whose total kept rank sets the budget")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocator", choices=("waterfill", "tau"), default="waterfill",
                        help="waterfill minimises sum_l P_l^2; tau equalises max_l P_l")
    parser.add_argument("--min-rank", type=int, default=0,
                        help="floor every module at this rank before allocating the rest")
    parser.add_argument("--copy-ranks", action="store_true",
                        help="reuse --budget-from's per-module ranks; isolates the "
                             "allocator from this script's own SVD round-trip noise")
    parser.add_argument("--budget-all", action="store_true",
                        help="keep every direction; a no-op rebuild that must score D_a ~ 0")
    args = parser.parse_args()

    cfg = json.loads((args.reference / "adapter_config.json").read_text())
    stored_r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / (math.sqrt(stored_r) if cfg.get("use_rslora") else stored_r)
    source_r = stored_r // 2

    weights = next(args.reference.glob("*.safetensors"))
    with safe_open(weights, framework="pt", device="cpu") as h:
        metadata = h.metadata()
        keys = set(h.keys())
        tensors = {k: h.get_tensor(k) for k in keys}

    # Budget: how many singular directions the energy-truncated variant kept.
    budget_weights = next(args.budget_from.glob("*.safetensors"))
    budget, source_ranks = 0, {}
    with safe_open(budget_weights, framework="pt", device="cpu") as h:
        for a_key in sorted(k for k in h.keys() if ".lora_A." in k):
            a = h.get_tensor(a_key)
            head = a.reshape(a.shape[0], -1)[:source_r]
            source_ranks[a_key] = int((head.float().norm(dim=1) > 1e-8).sum())
            budget += source_ranks[a_key]
    print(f"budget from {args.budget_from.name}: {budget} singular directions", flush=True)

    # Base weight norms.
    norms = {}
    for shard in sorted(args.base.rglob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as h:
            for key in h.keys():
                norms[key] = float(h.get_tensor(key).float().norm())

    # Per module: singular triplets of the original update, scored by the damage
    # dropping them would do relative to the layer's own weight.
    spectra, factors_of = {}, {}
    pool = []
    for a_key in sorted(k for k in keys if ".lora_A." in k):
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if b_key not in keys:
            continue
        a_full, b_full = tensors[a_key], tensors[b_key]
        a2, b2 = as_matrix(a_full, 0), as_matrix(b_full, 1)
        # The correction layout is head = +delta_trunc, tail = -delta_orig, so the
        # tail must be negated before it can be treated as the original update.
        a_orig, b_orig = a2[source_r:], -b2[:, source_r:]
        w = norms.get(base_name(a_key))
        if not w:
            continue
        u, s, vh = svd_of_product(a_orig, b_orig * scale)
        spectra[a_key] = (u, s, vh, w)
        factors_of[a_key] = (a_full, b_full)
        for i, sv in enumerate(s.tolist()):
            pool.append((sv * sv / (w * w), a_key, i))

    if args.copy_ranks:
        keep_counts = {k: source_ranks.get(k, 0) for k in spectra}
    elif args.budget_all:
        keep_counts = {k: len(v[1]) for k, v in spectra.items()}
    elif args.allocator == "tau":
        # Equalise the tail: k_l = min k with sqrt(sum_{i>k} sigma_i^2)/||W_l|| <= eps,
        # which is exactly S_l * sqrt(1 - tau_l) <= eps.  Bisect eps for the budget.
        tails = {}
        for a_key, (_, s, _, w) in spectra.items():
            sq = (s * s).flip(0).cumsum(0).flip(0)               # sum_{i>=k} sigma_i^2
            tails[a_key] = torch.cat([sq, sq.new_zeros(1)]).sqrt() / w
        def ranks_at(eps):
            return {k: int((v > eps).sum()) for k, v in tails.items()}
        lo, hi = 0.0, max(float(v[0]) for v in tails.values())
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if sum(ranks_at(mid).values()) > budget:
                lo = mid
            else:
                hi = mid
        keep_counts = ranks_at(hi)
        print(f"tau allocator: eps={hi:.5f}", flush=True)
    else:
        keep_counts = {k: 0 for k in spectra}
        pool.sort(reverse=True)
        for _, a_key, _ in pool[:budget]:
            keep_counts[a_key] += 1

    if args.min_rank and not args.budget_all:
        # Re-run the allocation with a floor, taking the floor out of the budget
        # first so the parameter count still matches.
        floor = {k: min(args.min_rank, len(v[1])) for k, v in spectra.items()}
        left = budget - sum(floor.values())
        keep_counts = dict(floor)
        rest = [(sc, k, i) for sc, k, i in pool if i >= floor[k]]
        rest.sort(reverse=True)
        for _, a_key, _ in rest[:max(left, 0)]:
            keep_counts[a_key] += 1
    kept = list(keep_counts.values())
    kept.sort()
    print(f"allocated ranks: min={kept[0]} p10={kept[len(kept)//10]} "
          f"median={kept[len(kept)//2]} p90={kept[int(0.9*len(kept))]} max={kept[-1]} "
          f"mean={sum(kept)/len(kept):.1f}", flush=True)

    out_tensors = {k: v for k, v in tensors.items() if ".lora_" not in k}
    checked = False
    for a_key, (u, s, vh, w) in spectra.items():
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        a_full, b_full = factors_of[a_key]
        k = keep_counts[a_key]
        root = s[:k].sqrt()
        new_b = (u[:, :k] * root[None, :]) / scale        # scale is reapplied by PEFT
        new_a = vh[:k, :] * root[:, None]
        a2, b2 = as_matrix(a_full, 0), as_matrix(b_full, 1)
        head_a = torch.zeros_like(a2[:a2.shape[0] // 2])
        head_b = torch.zeros_like(b2[:, :b2.shape[1] // 2])
        head_a[:k] = new_a
        head_b[:, :k] = new_b
        merged_a = torch.cat([head_a, a2[a2.shape[0] // 2:]], dim=0)
        merged_b = torch.cat([head_b, b2[:, b2.shape[1] // 2:]], dim=1)
        out_tensors[a_key] = merged_a.reshape(a_full.shape).to(a_full.dtype)
        out_tensors[b_key] = merged_b.reshape(b_full.shape).to(b_full.dtype)
        if k and not checked:
            dh = (head_b[:, :k] @ head_a[:k]) * scale
            dt = (b2[:, b2.shape[1] // 2:] @ a2[a2.shape[0] // 2:]) * scale
            cos = float((dh * dt).sum() / (dh.norm() * dt.norm() + 1e-12))
            print(f"sign check on {a_key.split('.')[-3]}: cos(head, tail) = {cos:+.4f} "
                  f"(must be negative)", flush=True)
            assert cos < 0, "rebuilt head has the wrong sign against the stored tail"
            checked = True

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, args.output / "adapter_model.safetensors", metadata=metadata)
    (args.output / "adapter_config.json").write_text(
        (args.reference / "adapter_config.json").read_text())
    (args.output / "allocation.json").write_text(json.dumps(
        {"budget": budget, "ranks": {k: v for k, v in keep_counts.items()}}, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
