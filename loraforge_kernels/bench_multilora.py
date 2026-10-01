#!/usr/bin/env python3
"""Does compressing to heterogeneous ranks actually pay in multi-LoRA serving?

Single-adapter inference amortises A and B over the whole batch, so its cost
barely tracks the rank -- measured all day, and it is why r=256 -> r=56 bought
under 2% end to end.  Multi-tenant serving is structurally different: every
token uses its own adapter, the LoRA weights are streamed per adapter, and the
traffic is proportional to the ranks present in the batch.

So this asks three separate questions:

  1. how large is the multi-LoRA sidecar relative to the base layer at all?
  2. does compressing 64 -> heterogeneous {4,8,16,32} shrink it?
  3. or does a uniform-rank kernel pad it all back to r_max and waste the
     compression -- the "runtime fails to realise the FLOP reduction" case?

Traffic distribution follows the multi-tenant protocol used by Punica and
S-LoRA: identical / uniform / skewed (Zipf) adapter popularity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from loraforge_kernels.multilora import (
    shrink_bucketed, shrink_grouped, shrink_padded, shrink_ragged,
)

K = 4096                  # llama-7b hidden width
REPS = 20                 # kernels chained inside one graph, to amortise its floor
DISTRIBUTIONS = ("identical", "uniform", "zipf1.0")
RANK_MIXES = [
    ("uniform r64", [64]),
    ("uniform r16", [16]),
    ("hetero {4,8,16,32}", [4, 8, 16, 32]),
]


def assign(n_adapters, tokens, dist, rng):
    if dist == "identical":
        return np.zeros(tokens, dtype=np.int64)
    if dist == "uniform":
        return rng.integers(0, n_adapters, tokens)
    alpha = float(dist.replace("zipf", ""))
    weights = 1.0 / np.arange(1, n_adapters + 1) ** alpha
    return rng.choice(n_adapters, size=tokens, p=weights / weights.sum())


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
    best = float("inf")
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    for _ in range(rounds):
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record(); g.replay(); e.record(); e.synchronize()
        best = min(best, b.elapsed_time(e))
    return best * 1000 / REPS


def build(n_adapters, ranks, dev, dtype, rng):
    """Give each adapter a rank drawn from the mix, then build both layouts."""
    per_adapter = [ranks[i % len(ranks)] for i in range(n_adapters)]
    r_max = max(per_adapter)
    a_pad = torch.randn(n_adapters, r_max, K, device=dev, dtype=dtype) / K**0.5
    for i, r in enumerate(per_adapter):
        a_pad[i, r:].zero_()          # the padding a uniform-rank kernel still reads
    # Ragged layout: the same weights end to end, plus offset and rank tables.
    a_flat = torch.cat([a_pad[i, :r] for i, r in enumerate(per_adapter)]).contiguous()
    offs, run = [], 0
    for r in per_adapter:
        offs.append(run); run += r
    return (per_adapter, r_max, a_pad, a_flat,
            torch.tensor(offs, device=dev), torch.tensor(per_adapter, device=dev))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--adapters", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/multilora.json"))
    args = parser.parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}   K={K}")
    rows = []

    for tokens in args.tokens:
        x = torch.randn(tokens, K, device=dev, dtype=dtype) / K**0.5
        # The base projection this sidecar rides on, for scale.
        w = torch.randn(K, K, device=dev, dtype=dtype) / K**0.5
        base_us = time_graphed(lambda: torch.mm(x, w.t()))
        print(f"\n=== tokens={tokens}   base q-projection {base_us:.1f} us")
        for n_adapters in args.adapters:
            print(f"  -- {n_adapters} adapters --")
            print(f"    {'rank mix':20s} {'traffic':>9s} {'dist':>9s} "
                  f"{'padded':>9s} {'grouped':>9s} {'ragged':>9s} {'bucketed':>9s} {'saved':>7s}")
            for mix_name, ranks in RANK_MIXES:
                per_adapter, r_max, a_pad, a_flat, offs, rank_tbl = build(
                    n_adapters, ranks, dev, dtype, rng)
                for dist in DISTRIBUTIONS:
                    order = assign(n_adapters, tokens, dist, rng)
                    idx = torch.tensor(order, device=dev)
                    # Bytes the batch must stream: padded reads r_max per token,
                    # grouped reads only each token's own rank.
                    pad_bytes = tokens * r_max * K * 2
                    grp_bytes = sum(per_adapter[i] for i in order) * K * 2

                    groups = {}
                    for rank in sorted(set(per_adapter)):
                        members = [i for i, r in enumerate(per_adapter) if r == rank]
                        remap = {a: j for j, a in enumerate(members)}
                        pos = [t for t, a in enumerate(order) if per_adapter[a] == rank]
                        if not pos:
                            continue
                        stack = torch.stack([a_pad[a, :rank] for a in members]).contiguous()
                        groups[rank] = (
                            stack,
                            torch.tensor(pos, device=dev),
                            torch.tensor([remap[order[t]] for t in pos], device=dev),
                        )

                    t_pad = time_graphed(lambda: shrink_padded(x, a_pad, idx))
                    t_grp = time_graphed(lambda: shrink_grouped(x, groups, idx))
                    t_rag = time_graphed(
                        lambda: shrink_ragged(x, a_flat, offs, rank_tbl, idx, r_max))
                    t_bkt = time_graphed(lambda: shrink_bucketed(x, groups, r_max))
                    # The ragged path must agree with the padded one it replaces.
                    ref = shrink_padded(x, a_pad, idx)
                    err = max(
                        float((shrink_ragged(x, a_flat, offs, rank_tbl, idx, r_max) - ref)
                              .abs().max() / ref.abs().max().clamp_min(1e-6)),
                        float((shrink_bucketed(x, groups, r_max) - ref)
                              .abs().max() / ref.abs().max().clamp_min(1e-6)),
                    )
                    flag = "" if err < 5e-2 else f"  !!rel_err={err:.1e}"
                    print(f"    {mix_name:20s} {grp_bytes/pad_bytes:8.2f}x {dist:>9s} "
                          f"{t_pad:9.1f} {t_grp:9.1f} {t_rag:9.1f} {t_bkt:9.1f} "
                          f"{100*(1-t_bkt/t_pad):6.1f}%{flag}")
                    rows.append({"tokens": tokens, "adapters": n_adapters, "mix": mix_name,
                                 "dist": dist, "r_max": r_max, "base_us": base_us,
                                 "padded_us": t_pad, "grouped_us": t_grp,
                                 "ragged_us": t_rag, "rel_err": err,
                                 "bucketed_us": t_bkt,
                                 "ragged_pct_of_base": 100 * t_rag / base_us,
                                 "bucketed_pct_of_base": 100 * t_bkt / base_us,
                                 "traffic_ratio": grp_bytes / pad_bytes,
                                 "padded_pct_of_base": 100 * t_pad / base_us,
                                 "grouped_pct_of_base": 100 * t_grp / base_us})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")
    print("\n=== does compression reach the wall clock? (uniform traffic, 32 adapters) ===")
    for tokens in args.tokens:
        sel = [r for r in rows if r["tokens"] == tokens and r["adapters"] == 32
               and r["dist"] == "uniform"]
        if not sel:
            continue
        by = {r["mix"]: r for r in sel}
        print(f"  tokens={tokens}  (sidecar as % of the base projection)")
        for mix, _ in RANK_MIXES:
            if mix in by:
                print(f"    {mix:20s} padded {by[mix]['padded_pct_of_base']:6.1f}%   "
                      f"grouped {by[mix]['grouped_pct_of_base']:6.1f}%   "
                      f"ragged {by[mix]['ragged_pct_of_base']:6.1f}%   "
                      f"bucketed {by[mix]['bucketed_pct_of_base']:6.1f}%")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
