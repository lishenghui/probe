#!/usr/bin/env python3
"""The whole multi-LoRA sidecar: shrink + expand, padded vs rank-bucketed.

Established so far, on the shrink alone:
  * in multi-tenant serving the sidecar tracks the rank (uniform r64 56us vs
    r16 16us) -- unlike single-adapter inference, where it does not;
  * heterogeneous ranks are half wasted by padding every adapter to r_max;
  * bucketing by rank recovers that only if the buckets run *concurrently*,
    because each bucket walks the whole K dimension and four of them in series
    cost more than one padded kernel;
  * concurrent bucketing then saves ~30% at 2048 tokens, at any rank mix.

The expand has the same per-token adapter indexing but a far larger output, so
it may not behave the same way.  This measures both halves and their sum,
against the base projection they ride on.

Each rank mix is measured twice: with the batch in arrival order, and with it
sorted so a bucket's tokens are contiguous.  Bucketing scatters which rows each
kernel touches, which costs nothing on the shrink's tiny [T, r] output but may
matter on the expand's [T, N_out] one.  Sorting the batch by adapter is what
segmented multi-LoRA runtimes already do, so it is free at the scheduler level
-- the question is whether it is *required*.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from loraforge_kernels.multilora import (
    expand_bucketed_parallel, expand_padded,
    shrink_bucketed_parallel, shrink_padded,
)

K = N_OUT = 4096
RANKS = [4, 8, 16, 32]
CONCENTRATIONS = [1.0, 0.5, 0.25]
TOKENS = [512, 2048]
SPLIT_KS = [1, 2, 4, 8]
REPS = 20
N_ADAPTERS = 64


def time_graphed(fn, warmup=8, rounds=12):
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


def best_over(make_fn, options):
    best, arg = float("inf"), None
    for opt in options:
        try:
            t = time_graphed(make_fn(opt))
        except Exception:
            continue
        if t < best:
            best, arg = t, opt
    return best, arg



def run_one(layout, order, conc, x, y, base_us, a_pad, b_pad,
            per_adapter, by_rank, r_max, dev, streams, events, rows):
    """One rank mix under one batch layout."""
    if layout == "sorted":
        order = order[np.argsort([per_adapter[a] for a in order], kind="stable")]
    tokens = len(order)
    idx = torch.tensor(order, device=dev)

    buckets_a, buckets_b = {}, {}
    for r in sorted(by_rank):
        members = by_rank[r]
        remap = {a: j for j, a in enumerate(members)}
        pos = [t for t, a in enumerate(order) if per_adapter[a] == r]
        if not pos:
            continue
        p_t = torch.tensor(pos, device=dev)
        l_t = torch.tensor([remap[order[t]] for t in pos], device=dev)
        buckets_a[r] = (torch.stack([a_pad[a, :r] for a in members]).contiguous(), p_t, l_t)
        buckets_b[r] = (torch.stack([b_pad[a, :, :r] for a in members]).contiguous(), p_t, l_t)

    z_ref = shrink_padded(x, a_pad, idx, split_k=1)
    z_bkt = shrink_bucketed_parallel(x, buckets_a, r_max, streams, events, split_k=1)
    err = float((z_bkt - z_ref).abs().max() / z_ref.abs().max().clamp_min(1e-6))
    zb = z_ref.to(x.dtype)
    e_ref = expand_padded(y, zb, b_pad, idx)
    e_bkt = expand_bucketed_parallel(y, zb, buckets_b, streams, events)
    err = max(err, float((e_bkt - e_ref).abs().max() / e_ref.abs().max().clamp_min(1e-6)))

    t_sp, _ = best_over(lambda sk: (lambda: shrink_padded(x, a_pad, idx, split_k=sk)), SPLIT_KS)
    t_sb, _ = best_over(lambda sk: (lambda: shrink_bucketed_parallel(
        x, buckets_a, r_max, streams, events, split_k=sk)), SPLIT_KS)
    t_ep, _ = best_over(lambda bn: (lambda: expand_padded(y, zb, b_pad, idx, bn=bn)), [64, 128, 256])
    t_eb, _ = best_over(lambda bn: (lambda: expand_bucketed_parallel(
        y, zb, buckets_b, streams, events, bn=bn)), [64, 128, 256])
    tot_p, tot_b = t_sp + t_ep, t_sb + t_eb
    flag = "" if err < 5e-2 else f"  !!rel_err={err:.1e}"
    print(f"    {conc:5.2f} {layout:8s} | {t_sp:9.1f} {t_sb:9.1f} | {t_ep:9.1f} {t_eb:9.1f} | "
          f"{tot_p:9.1f} {tot_b:9.1f} {100*(1-tot_b/tot_p):6.1f}% | "
          f"{100*tot_p/base_us:4.0f}% -> {100*tot_b/base_us:3.0f}%{flag}")
    rows.append({"tokens": tokens, "concentration": conc, "layout": layout,
                 "base_us": base_us, "shrink_padded_us": t_sp, "shrink_bucketed_us": t_sb,
                 "expand_padded_us": t_ep, "expand_bucketed_us": t_eb,
                 "total_padded_us": tot_p, "total_bucketed_us": tot_b,
                 "saved_pct": 100 * (1 - tot_b / tot_p), "rel_err": err})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/multilora_full.json"))
    args = parser.parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    r_max = max(RANKS)
    print(f"GPU: {torch.cuda.get_device_name(0)}  K={K} N_out={N_OUT} "
          f"ranks={RANKS} adapters={N_ADAPTERS}")

    per_adapter = [RANKS[i % len(RANKS)] for i in range(N_ADAPTERS)]
    a_pad = torch.randn(N_ADAPTERS, r_max, K, device=dev, dtype=dtype) / K**0.5
    b_pad = torch.randn(N_ADAPTERS, N_OUT, r_max, device=dev, dtype=dtype) / r_max**0.5
    for i, r in enumerate(per_adapter):
        a_pad[i, r:].zero_()
        b_pad[i, :, r:].zero_()
    by_rank = {r: [i for i, rr in enumerate(per_adapter) if rr == r] for r in RANKS}

    streams = [torch.cuda.Stream() for _ in RANKS]
    events = [torch.cuda.Event() for _ in range(len(RANKS) + 1)]
    rows = []

    for tokens in TOKENS:
        x = torch.randn(tokens, K, device=dev, dtype=dtype) / K**0.5
        w = torch.randn(N_OUT, K, device=dev, dtype=dtype) / K**0.5
        y = torch.randn(tokens, N_OUT, device=dev, dtype=dtype)
        base_us = time_graphed(lambda: torch.mm(x, w.t()))
        print(f"\n=== tokens={tokens}   base projection {base_us:.1f} us")
        print(f"    {'conc':>5s} {'layout':8s} | {'shr pad':>9s} {'shr bkt':>9s} | "
              f"{'exp pad':>9s} {'exp bkt':>9s} | {'tot pad':>9s} "
              f"{'tot bkt':>9s} {'saved':>7s} | {'% of base':>10s}")
        for conc in CONCENTRATIONS:
            dominant, n_dom = 16, int(round(tokens * conc))
            others = [r for r in RANKS if r != dominant]
            order = list(rng.choice(by_rank[dominant], n_dom))
            for j in range(tokens - n_dom):
                order.append(int(rng.choice(by_rank[others[j % len(others)]])))
            rng.shuffle(order)
            order = np.array(order[:tokens], dtype=np.int64)
            for layout in ("arrival", "sorted"):
                run_one(layout, order, conc, x, y, base_us, a_pad, b_pad,
                        per_adapter, by_rank, r_max, dev, streams, events, rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
