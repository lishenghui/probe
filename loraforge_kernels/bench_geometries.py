#!/usr/bin/env python3
"""Which model geometry would actually show rank compression end to end?

On Wan2.1 the sidecar is only ~5% of a generation, so no energy threshold can
buy much throughput.  The block sweep says the sidecar is 27-40% of the *linear
stack*; what dilutes it is attention, which is 53-56% of Wan's device time at
32760 tokens.

This times a full transformer block -- LoRA'd projections plus attention plus
the norms -- for several real geometries, so the sidecar's share of something
comparable to end-to-end is visible before committing to a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from loraforge_kernels.fused_linear import (
    _concat, _hybrid, _kconcat, _torch_naive,
    augment_bias, augment_weight, kconcat_weight, packed_rank,
)

# name, dim, ffn, n_heads, n_kv_heads, tokens, note
# Wan at several clip lengths: attention is O(M^2) and the linears are O(M), so
# shortening the clip should collapse attention's share and lift the sidecar's.
GEOMETRIES = [
    ("wan2.1 480p 81f", 1536, 8960, 12, 12, 32760, "what we measured: attn dominates"),
    ("wan2.1 480p 21f", 1536, 8960, 12, 12, 8580, "shorter clip"),
    ("wan2.1 480p 5f", 1536, 8960, 12, 12, 3120, "shortest clip"),
    ("wan2.1 256p 21f", 1536, 8960, 12, 12, 2688, "small frames, short clip"),
    ("sdxl-unet mid @1024", 1280, 5120, 20, 20, 4096, "image diffusion"),
]
RANKS = [256, 144, 56]


def time_fn(fn, warmup=5, rounds=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record(); fn(); e.record(); e.synchronize()
        best = min(best, b.elapsed_time(e))
    return best


def best_linear(x, w, bias, a, b, scale):
    """Cheapest correct sidecar for this shape, as the selector would pick."""
    n, k = w.shape
    cands = [lambda: _hybrid(x, w, bias, a, b, scale)]
    w_aug, b_aug = augment_weight(w, a), augment_bias(bias, a.shape[0])
    cands.append(lambda: _concat(x, w_aug, b_aug, b, scale, n))
    if 3 * k < 2 * n:
        w_kc = kconcat_weight(w, b)
        cands.append(lambda: _kconcat(x, w_kc, bias, a, scale, packed_rank(a.shape[0])))
    return min(time_fn(c) for c in cands)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/geometries.json"))
    args = parser.parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    rows = []

    for name, dim, ffn, heads, kv_heads, m, note in GEOMETRIES:
        hd = dim // heads
        kv_dim = kv_heads * hd
        x = torch.randn(m, dim, device=dev, dtype=dtype) / dim**0.5
        # (out_features, label) for every LoRA'd projection in one block.
        projs = [(dim, "q"), (kv_dim, "k"), (kv_dim, "v"), (dim, "o"),
                 (ffn, "gate"), (ffn, "up"), (dim, "down")]

        # Attention, at the shape this geometry implies.
        q = torch.randn(1, heads, m, hd, device=dev, dtype=dtype)
        kk = torch.randn(1, kv_heads, m, hd, device=dev, dtype=dtype)
        vv = torch.randn(1, kv_heads, m, hd, device=dev, dtype=dtype)
        attn_us = time_fn(lambda: F.scaled_dot_product_attention(
            q, kk, vv, enable_gqa=(kv_heads != heads))) * 1000

        base_us = 0.0
        for out, _ in projs:
            k_in = ffn if out == dim and _ == "down" else dim
            w = torch.randn(out, k_in, device=dev, dtype=dtype) / k_in**0.5
            xi = x if k_in == dim else torch.randn(m, k_in, device=dev, dtype=dtype) / k_in**0.5
            base_us += time_fn(lambda: F.linear(xi, w)) * 1000

        print(f"=== {name}  dim={dim} ffn={ffn} tokens={m}   ({note})")
        print(f"    linears {base_us:8.1f} us | attention {attn_us:8.1f} us "
              f"| attn share {100*attn_us/(base_us+attn_us):5.1f}%")
        for rank in RANKS:
            sidecar_us = 0.0
            for out, label in projs:
                k_in = ffn if (out == dim and label == "down") else dim
                xi = x if k_in == dim else torch.randn(m, k_in, device=dev, dtype=dtype) / k_in**0.5
                w = torch.randn(out, k_in, device=dev, dtype=dtype) / k_in**0.5
                bias = None
                a = torch.randn(rank, k_in, device=dev, dtype=dtype) / k_in**0.5
                b = torch.randn(out, rank, device=dev, dtype=dtype) / rank**0.5
                plain = time_fn(lambda: F.linear(xi, w)) * 1000
                sidecar_us += best_linear(xi, w, bias, a, b, 1.0) * 1000 - plain
            block = base_us + attn_us
            print(f"    rank {rank:4d}: sidecar {sidecar_us:8.1f} us = "
                  f"{100*sidecar_us/block:5.1f}% of the block")
            rows.append({"geometry": name, "dim": dim, "ffn": ffn, "tokens": m,
                         "rank": rank, "linears_us": base_us, "attn_us": attn_us,
                         "sidecar_us": sidecar_us,
                         "sidecar_pct_of_block": 100 * sidecar_us / block})
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")

    print("=== sidecar share of a full block, and what compressing 256 -> 56 buys ===")
    for name, *_ in GEOMETRIES:
        hits = {r["rank"]: r for r in rows if r["geometry"] == name}
        if len(hits) < 3:
            continue
        hi, lo = hits[256]["sidecar_pct_of_block"], hits[56]["sidecar_pct_of_block"]
        block = hits[256]["linears_us"] + hits[256]["attn_us"]
        gain = 100 * ((block + hits[256]["sidecar_us"]) / (block + hits[56]["sidecar_us"]) - 1)
        print(f"  {name:26s} sidecar {hi:5.1f}% -> {lo:5.1f}%   throughput gain {gain:+5.1f}%")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
