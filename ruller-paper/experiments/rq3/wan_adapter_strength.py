#!/usr/bin/env python3
"""Adapter strength and spectral loss for public Wan2.1-T2V-1.3B LoRAs.

This is the cheap screen that decides whether a video pool is worth evaluating.
Video evaluation costs orders of magnitude more than text -- one variant is a
batch of generated clips plus a VBench or FVD pass -- so it is only worth paying
if the population actually spans a range of adapter strengths. A pool whose S is
homogeneous cannot exhibit the effect this paper is about, however clean its
evaluation protocol is.

Nothing here needs a GPU: the spectra are r x r cores with r <= 128 over ~300
modules, and the cost is dominated by reading the 5 GB base checkpoint.

The pool uses two key conventions and therefore two packagings of the same base:

  native Wan      blocks.0.self_attn.q.lora_A.default.weight  -> blocks.0.self_attn.q.weight
  PEFT/Diffusers  base_model.model.blocks.0.attn1.to_k.lora_A.weight
                                                              -> blocks.0.attn1.to_k.weight
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def lora_pairs(weights: dict) -> list[tuple[str, str, str]]:
    """(A key, B key, base module name) for every LoRA pair present."""
    out = []
    for a in sorted(weights):
        if ".lora_A" not in a and ".lora_down" not in a:
            continue
        b = (a.replace(".lora_A", ".lora_B") if ".lora_A" in a
             else a.replace(".lora_down", ".lora_up"))
        if b not in weights:
            continue
        base = re.sub(r"\.lora_[AB](\.default)?\.weight$", ".weight", a)
        base = re.sub(r"\.lora_(down|up)(\.default)?\.weight$", ".weight", base)
        for prefix in ("base_model.model.", "base_model.", "diffusion_model."):
            if base.startswith(prefix):
                base = base[len(prefix):]
        out.append((a, b, base))
    return out


def spectrum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a32, b32 = a.float(), b.float()
    if a32.ndim > 2:
        a32 = a32.reshape(a32.shape[0], -1)
    if b32.ndim > 2:
        b32 = b32.reshape(b32.shape[0], b32.shape[1])
    qb, rb = torch.linalg.qr(b32, mode="reduced")
    qa, ra = torch.linalg.qr(a32.T, mode="reduced")
    return torch.linalg.svdvals(rb @ ra.T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", type=Path, required=True,
                    help="json list of {name, path, base, scale}")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.99, 0.95, 0.90, 0.80, 0.70, 0.50])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.adapters.read_text())
    norms_cache: dict[str, dict[str, float]] = {}
    results = []
    for entry in spec:
        base_dir = Path(entry["base"])
        if entry["base"] not in norms_cache:
            # lazy per-tensor reads: loading the whole 5 GB checkpoint to take a few
            # hundred Frobenius norms is what made the first version time out
            index = {}
            for f in sorted(base_dir.glob("*.safetensors")):
                with safe_open(f, framework="pt") as h:
                    for k in h.keys():
                        index[k] = f
            norms_cache[entry["base"]] = {"__index__": index}
            print(f"  base {base_dir.parent.name}: {len(index)} tensors indexed",
                  flush=True)
        cache = norms_cache[entry["base"]]
        index = cache["__index__"]

        def base_norm(key):
            if key in cache:
                return cache[key]
            f = index.get(key)
            if f is None:
                return None
            with safe_open(f, framework="pt") as h:
                cache[key] = float(h.get_tensor(key).float().norm())
            return cache[key]

        p = Path(entry["path"])
        w = (load_file(p) if p.suffix == ".safetensors"
             else torch.load(p, map_location="cpu", weights_only=False))
        if isinstance(w, dict) and "state_dict" in w:
            w = w["state_dict"]
        scale = entry.get("scale", 1.0)

        num = den = 0.0
        spectra, matched, missing = {}, 0, []
        for a, b, base in lora_pairs(w):
            n = base_norm(base)
            if n is None:
                missing.append(base)
                continue
            sv = spectrum(w[a], w[b]) * scale
            spectra[a] = sv
            num += float(sv.square().sum())
            den += n * n
            matched += 1
        if not spectra:
            print(f"{entry['name']}: no matched modules; first misses {missing[:3]}",
                  flush=True)
            continue
        S = math.sqrt(num / den)
        rank = max(sv.numel() for sv in spectra.values())
        rec = {"name": entry["name"], "S": S, "modules": matched,
               "unmatched": len(missing), "rank": rank, "scale": scale, "L_W": {}}
        for tau in args.thresholds:
            kept = drop = 0.0
            k_tot = n_tot = 0
            for sv in spectra.values():
                e = sv.square()
                c = torch.cumsum(e, 0)
                k = min(int(torch.searchsorted(c, tau * c[-1]).item()) + 1, e.numel())
                kept += float(e[:k].sum())
                drop += float(e[k:].sum())
                k_tot += k
                n_tot += e.numel()
            label = f"e{round(tau * 100):02d}"
            rec["L_W"][label] = math.sqrt(drop / (kept + drop)) if kept + drop else 0.0
            rec.setdefault("rank_frac", {})[label] = k_tot / n_tot
        results.append(rec)
        print(f"{entry['name']:12s} S={S:.4f} rank={rank:4d} modules={matched:4d} "
              f"(unmatched {len(missing)})  L_W@.90={rec['L_W']['e90']:.3f}  "
              f"rank_frac@.90={rec['rank_frac']['e90']:.2f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(results)} adapters)")


if __name__ == "__main__":
    main()
