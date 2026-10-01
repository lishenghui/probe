#!/usr/bin/env python3
"""When does rank-aware batching pay? Sweep the batch's rank concentration.

Established in bench_multilora.py: in multi-tenant serving the sidecar is
1.6-5.7x a base projection and its cost really does track the rank (uniform
r64 56us vs r16 16us).  Compression produces *heterogeneous* ranks, a uniform
kernel pads them all to r_max and throws half that saving away, and bucketing by
rank recovers it -- but only when the batch's ranks are concentrated.  With the
tokens spread evenly over four rank buckets, each bucket is too small to fill
the GPU and bucketing loses.

So the deciding variable is not the kernel, it is how concentrated the batch is,
which is a scheduling choice.  This sweeps that concentration against the batch
size to find the crossover, i.e. the rule a rank-aware scheduler would need.

`concentration` is the fraction of tokens whose adapter carries the dominant
rank; the remainder is spread over the other ranks.  1.0 is a rank-homogeneous
batch, 0.25 is fully mixed over four ranks.

Both paths get their own best SPLIT_K from a sweep rather than a heuristic.  An
earlier version fixed the padded path at 8 while letting each bucket pick its
own, which over-split the small buckets and made bucketing look like it lost
under mixed traffic -- the artefact, not the effect.  A heuristic keyed on token
count is no better: it made a 512-token batch slower than a 2048-token one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from loraforge_kernels.multilora import (
    shrink_bucketed, shrink_bucketed_parallel, shrink_padded,
)

K = 4096
RANKS = [4, 8, 16, 32]
CONCENTRATIONS = [1.0, 0.9, 0.75, 0.5, 0.25]
TOKENS = [32, 128, 512, 2048]
REPS = 20
N_ADAPTERS = 64
SPLIT_KS = [1, 2, 4, 8, 16, 32]


def time_graphed(fn, warmup=8, rounds=15):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=torch.cuda.graphs.graph_pool_handle()):
        for _ in range(REPS):
            fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record(); g.replay(); e.record(); e.synchronize()
        best = min(best, b.elapsed_time(e))
    return best * 1000 / REPS


def best_over_splits(make_fn):
    """Cheapest SPLIT_K for this configuration, so the comparison is not a
    comparison of two different split factors."""
    best, best_sk = float("inf"), None
    for sk in SPLIT_KS:
        try:
            t = time_graphed(make_fn(sk))
        except Exception:
            continue
        if t < best:
            best, best_sk = t, sk
    return best, best_sk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/rank_affinity.json"))
    args = parser.parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    r_max = max(RANKS)
    print(f"GPU: {torch.cuda.get_device_name(0)}   K={K}  ranks={RANKS}  adapters={N_ADAPTERS}")

    # Adapters cycle through the rank mix; the batch decides which ones get used.
    per_adapter = [RANKS[i % len(RANKS)] for i in range(N_ADAPTERS)]
    a_pad = torch.randn(N_ADAPTERS, r_max, K, device=dev, dtype=dtype) / K**0.5
    for i, r in enumerate(per_adapter):
        a_pad[i, r:].zero_()
    by_rank = {r: [i for i, rr in enumerate(per_adapter) if rr == r] for r in RANKS}

    # Streams and events for the concurrent-bucket path must outlive graph
    # capture, so allocate them once.
    streams = [torch.cuda.Stream() for _ in range(len(RANKS))]
    events = [torch.cuda.Event() for _ in range(len(RANKS) + 1)]

    rows = []
    for tokens in TOKENS:
        x = torch.randn(tokens, K, device=dev, dtype=dtype) / K**0.5
        w = torch.randn(K, K, device=dev, dtype=dtype) / K**0.5
        base_us = time_graphed(lambda: torch.mm(x, w.t()))
        # References: what a homogeneous batch at the mix's max and mean rank costs.
        hom = {}
        for r in (r_max, 16):
            stack = a_pad[by_rank[r_max][:1]].repeat(N_ADAPTERS, 1, 1)[:, :r].contiguous()
            idx_h = torch.tensor(rng.integers(0, N_ADAPTERS, tokens), device=dev)
            hom[r], _ = best_over_splits(
                lambda sk, s=stack, i=idx_h: (lambda: shrink_padded(x, s, i, split_k=sk)))
        print(f"\n=== tokens={tokens}  base proj {base_us:.1f} us "
              f"| homogeneous r{r_max} {hom[r_max]:.1f} us, r16 {hom[16]:.1f} us")
        print(f"    {'conc':>5s} {'buckets':>8s} {'min bkt':>8s} "
              f"{'padded':>9s} {'sk':>3s} {'serial':>9s} {'sk':>3s} {'parallel':>9s} {'sk':>3s} {'saved':>7s}  verdict")
        for conc in CONCENTRATIONS:
            dominant = 16
            n_dom = int(round(tokens * conc))
            others = [r for r in RANKS if r != dominant]
            order = []
            order += list(rng.choice(by_rank[dominant], n_dom))
            for j in range(tokens - n_dom):
                r = others[j % len(others)]
                order.append(int(rng.choice(by_rank[r])))
            rng.shuffle(order)
            order = np.array(order[:tokens], dtype=np.int64)
            idx = torch.tensor(order, device=dev)

            buckets, sizes = {}, {}
            for r in RANKS:
                members = [i for i, rr in enumerate(per_adapter) if rr == r]
                remap = {a: j for j, a in enumerate(members)}
                pos = [t for t, a in enumerate(order) if per_adapter[a] == r]
                if not pos:
                    continue
                sizes[r] = len(pos)
                buckets[r] = (
                    torch.stack([a_pad[a, :r] for a in members]).contiguous(),
                    torch.tensor(pos, device=dev),
                    torch.tensor([remap[order[t]] for t in pos], device=dev),
                )

            ref = shrink_padded(x, a_pad, idx, split_k=1)
            got = shrink_bucketed(x, buckets, r_max, split_k=1)
            err = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
            t_pad, sk_pad = best_over_splits(
                lambda sk: (lambda: shrink_padded(x, a_pad, idx, split_k=sk)))
            t_bkt, sk_bkt = best_over_splits(
                lambda sk: (lambda: shrink_bucketed(x, buckets, r_max, split_k=sk)))
            t_par, sk_par = best_over_splits(
                lambda sk: (lambda: shrink_bucketed_parallel(
                    x, buckets, r_max, streams, events, split_k=sk)))
            got_par = shrink_bucketed_parallel(x, buckets, r_max, streams, events, split_k=1)
            err = max(err, float((got_par - ref).abs().max() / ref.abs().max().clamp_min(1e-6)))
            saved = 100 * (1 - min(t_bkt, t_par) / t_pad)
            verdict = "bucket wins" if saved > 2 else ("pad wins" if saved < -2 else "tie")
            flag = "" if err < 5e-2 else f"  !!rel_err={err:.1e}"
            print(f"    {conc:5.2f} {len(buckets):8d} {min(sizes.values()):8d} "
                  f"{t_pad:9.1f} {sk_pad:3d} {t_bkt:9.1f} {sk_bkt:3d} "
                  f"{t_par:9.1f} {sk_par:3d} {saved:6.1f}%  {verdict}{flag}")
            rows.append({"tokens": tokens, "concentration": conc, "buckets": len(buckets),
                         "min_bucket": min(sizes.values()), "base_us": base_us,
                         "padded_us": t_pad, "bucketed_us": t_bkt, "saved_pct": saved,
                         "split_k_padded": sk_pad, "split_k_bucketed": sk_bkt,
                         "parallel_us": t_par, "split_k_parallel": sk_par,
                         "homogeneous_rmax_us": hom[r_max], "homogeneous_r16_us": hom[16],
                         "rel_err": err})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")

    print("\n=== crossover: lowest concentration at which bucketing still wins ===")
    for tokens in TOKENS:
        sel = sorted((r for r in rows if r["tokens"] == tokens), key=lambda r: -r["concentration"])
        win = [r["concentration"] for r in sel if r["saved_pct"] > 2]
        smallest = min(win) if win else None
        print(f"  tokens={tokens:5d}: " +
              (f"wins down to concentration {smallest:.2f} "
               f"(min bucket {min(r['min_bucket'] for r in sel if r['concentration'] == smallest)} tokens)"
               if smallest else "never wins"))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
