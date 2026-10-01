#!/usr/bin/env python3
"""Does Split-K make a small LoRA rank actually faster?

The claim under test: the shrink x[M,K] @ A[r,K].T has only M*r worth of output,
so at small rank a generic GEMM cannot fill the GPU -- FLOPs fall with the rank
but so does parallelism, and latency stays flat.  Splitting K across blocks
manufactures work units and should restore the rank->speed relationship.

Swept two-dimensionally over rank and split factor, at several M, against
cuBLAS.  Timed from a CUDA graph replay as well as eagerly: at small M the
launch cost is comparable to the kernel, and conflating the two is how this
question gets answered wrongly.

The number that decides it is not "is Split-K faster than cuBLAS" but "does
time fall as the rank falls" -- printed as the r=64 -> r=8 ratio per method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from loraforge_kernels.splitk_shrink import shrink_atomic, shrink_partial

RANKS = [4, 8, 16, 32, 64]
SPLITS = [1, 2, 4, 8, 16]
# (label, M, K) -- decode through prefill on a llama-7b width, plus a diffusion M.
CASES = [("decode M=1", 1, 4096), ("decode M=16", 16, 4096), ("batch M=64", 64, 4096),
         ("prefill M=512", 512, 4096), ("diffusion M=32760", 32760, 1536)]


def time_fn(fn, warmup=5, rounds=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record(); fn(); e.record(); e.synchronize()
        best = min(best, b.elapsed_time(e))
    return best * 1000 / REPS  # us per kernel


REPS = 50


def graphed(fn, warmup=12, reps=REPS):
    """Capture `reps` copies of the kernel in one graph.

    A single-kernel graph replay costs ~9 us on this machine whatever it
    contains, which is more than the shrink itself at these shapes -- timing one
    replay measures the replay, not the kernel.  Replaying a chain of `reps`
    launches and dividing amortises that floor away.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=torch.cuda.graphs.graph_pool_handle()):
        for _ in range(reps):
            fn()
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/splitk.json"))
    args = parser.parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    rows = []

    for label, m, k in CASES:
        x = torch.randn(m, k, device=dev, dtype=dtype) / k**0.5
        print(f"\n=== {label}  K={k} ===")
        print(f"{'rank':>5s} {'cuBLAS':>9s} | " +
              " ".join(f"{'p/s'+str(s):>9s}" for s in SPLITS) + " | " +
              " ".join(f"{'a/s'+str(s):>9s}" for s in SPLITS) + " | best")
        per_rank = {}
        for r in RANKS:
            a = torch.randn(r, k, device=dev, dtype=dtype) / k**0.5
            want = torch.mm(x, a.t())
            denom = want.abs().max().clamp_min(1e-6)
            cub = time_fn(graphed(lambda: torch.mm(x, a.t())).replay)
            line = f"{r:5d} {cub:9.1f} | "
            best, best_tag = cub, "cuBLAS"
            entry = {"cublas": cub}
            for fam, fn in (("p", shrink_partial), ("a", shrink_atomic)):
                for s in SPLITS:
                    try:
                        got = fn(x, a, split_k=s)
                        err = float((got.float() - want.float()).abs().max() / denom)
                        t = time_fn(graphed(lambda f=fn, ss=s: f(x, a, split_k=ss)).replay)
                    except Exception:
                        t, err = float("nan"), float("nan")
                    if err > 5e-2:
                        t = float("nan")
                    entry[f"{fam}{s}"] = t
                    line += f"{t:9.1f} "
                    if t == t and t < best:
                        best, best_tag = t, f"{fam}/s{s}"
                line += "| " if fam == "p" else ""
            print(line + f"| {best_tag} {best:.1f}us")
            entry["best"], entry["best_tag"] = best, best_tag
            per_rank[r] = entry
            rows.append({"case": label, "m": m, "k": k, "rank": r, **entry})

        print(f"  --- does time fall with rank?  (r=64 -> r=8) ---")
        for name in ["cublas", "best"]:
            hi, lo = per_rank[64][name], per_rank[8][name]
            print(f"    {name:8s} {hi:8.1f} -> {lo:8.1f} us   "
                  f"{'FASTER' if lo < hi else 'no gain'} {hi/lo:5.2f}x")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
