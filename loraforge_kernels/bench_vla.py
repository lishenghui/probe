#!/usr/bin/env python3
"""The VLA regime: does rank compression pay where decode dominates?

OpenVLA-7B is a llama2-7b backbone that sees ~280 prefill tokens (256 vision
patches plus a short instruction) and then autoregressively emits 7 action
tokens at batch 1.  That is the opposite of Wan: attention is negligible, the
linear layers are weight-streaming bound, and the LoRA adapter is rank 64
compressed to a mean retained rank of 27/19/14 at e99/e95/e90.

Two regimes matter and they fail differently:
  prefill (M=280)  -- skinny GEMMs, sidecar cost near its traffic floor
  decode  (M=1)    -- one GEMV per projection; the sidecar's own kernel launches
                      can outweigh every byte it moves, and launches do not care
                      about the rank at all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from loraforge_kernels.fused_linear import (
    _concat, _hybrid, _torch_naive, augment_bias, augment_weight,
)
from loraforge_kernels.fused_lora import fused_lora
from loraforge_kernels.splitk_lora import splitk_persistent_fused_lora

# llama2-7b backbone, one decoder layer's LoRA'd projections.
DIM, FFN, LAYERS = 4096, 11008, 32
PROJS = [("q", DIM, DIM), ("k", DIM, DIM), ("v", DIM, DIM), ("o", DIM, DIM),
         ("gate", DIM, FFN), ("up", DIM, FFN), ("down", FFN, DIM)]
# rank 64 adapter, and what FraQ retains at e99 / e95 / e90.
RANKS = [("full", 64), ("e99", 27), ("e95", 19), ("e90", 14)]


def capture(fn, warmup=12):
    """Replay the whole layer from a CUDA graph.

    If the sidecar's cost is launch and dispatch rather than memory or maths,
    replaying a captured graph removes it: one graph launch replaces every
    kernel launch and all the Python in between.  Warm up first so Triton
    autotuning and the allocator have settled before capture.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graphs.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        fn()
    return graph


def time_fn(fn, warmup=10, rounds=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record(); fn(); e.record(); e.synchronize()
        best = min(best, b.elapsed_time(e))
    return best


class Layer:
    """One decoder layer's worth of LoRA'd projections, timed as a unit.

    Timing the whole layer rather than each projection keeps per-call launch
    overhead in the measurement where it belongs -- it is the dominant term at
    M=1, and measuring projections one at a time would hide it.
    """

    def __init__(self, m, rank, dtype, dev):
        self.xs, self.w, self.a, self.b = [], [], [], []
        self.w_aug, self.b_aug, self.bias = [], [], []
        for _, k, n in PROJS:
            self.xs.append(torch.randn(m, k, device=dev, dtype=dtype) / k**0.5)
            self.w.append(torch.randn(n, k, device=dev, dtype=dtype) / k**0.5)
            self.bias.append(None)
            a = torch.randn(rank, k, device=dev, dtype=dtype) / k**0.5
            b = torch.randn(n, rank, device=dev, dtype=dtype) / rank**0.5
            self.a.append(a); self.b.append(b)
            self.w_aug.append(augment_weight(self.w[-1], a))
            self.b_aug.append(None)
        self.scale = 16.0 / rank
        # Merging is a load-time transform; timing it per call would measure the
        # transform rather than the inference it enables.
        self.w_merged = [w + self.scale * (b @ a)
                         for w, a, b in zip(self.w, self.a, self.b)]

    def base(self):
        return [F.linear(x, w) for x, w in zip(self.xs, self.w)]

    def merged(self):
        return [F.linear(x, w) for x, w in zip(self.xs, self.w_merged)]

    def torch_naive(self):
        return [_torch_naive(x, w, bi, a, b, self.scale)
                for x, w, bi, a, b in zip(self.xs, self.w, self.bias, self.a, self.b)]

    def hybrid(self):
        return [_hybrid(x, w, bi, a, b, self.scale)
                for x, w, bi, a, b in zip(self.xs, self.w, self.bias, self.a, self.b)]

    def concat(self):
        return [_concat(x, wa, ba, b, self.scale, w.shape[0])
                for x, wa, ba, b, w in zip(self.xs, self.w_aug, self.b_aug, self.b, self.w)]

    def row_kernel(self):
        """One launch fuses shrink, expand and the residual add -- but its grid is
        (M,), so at M=1 the whole sidecar runs on a single SM."""
        out = []
        for x, w, a, b in zip(self.xs, self.w, self.a, self.b):
            y = F.linear(x, w)
            out.append(fused_lora(x, a, b, y, self.scale))
        return out

    def splitk(self):
        """Split-K shrink plus an expand whose grid also covers N, so the decode
        sidecar is spread over the machine instead of one block."""
        out = []
        for x, w, a, b in zip(self.xs, self.w, self.a, self.b):
            y = F.linear(x, w)
            out.append(splitk_persistent_fused_lora(x, a, b, y, self.scale))
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/vla.json"))
    args = parser.parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"llama2-7b backbone: dim={DIM} ffn={FFN} layers={LAYERS}\n")
    rows = []

    for label, m in (("decode  M=1", 1), ("prefill M=280", 280)):
        print(f"=== {label} ===")
        strategies = ["merged", "torch_naive", "hybrid", "concat", "row_kernel", "splitk"]
        print(f"{'variant':8s} {'rank':>5s} {'base/layer':>11s} " +
              " ".join(f"{s[:10]:>11s}" for s in strategies) + "   best sidecar")
        for tag, rank in RANKS:
            layer = Layer(m, rank, dtype, dev)
            fns = {"base": layer.base}
            fns.update({s: getattr(layer, s) for s in strategies})
            # Correctness guard before timing anything.
            want = layer.torch_naive()
            for s in strategies:
                got = getattr(layer, s)()
                err = max(float((p.float() - q.float()).abs().max() / q.abs().max().clamp_min(1e-6))
                          for p, q in zip(got, want))
                if err > 5e-2:
                    print(f"  !! {s} rel_err={err:.2e}")
            t = {k: time_fn(f) for k, f in fns.items()}
            # And again from a captured graph, which costs one launch total.
            for k, f in list(fns.items()):
                try:
                    g = capture(f)
                    t[k + "|g"] = time_fn(g.replay)
                except Exception as exc:
                    print(f"  (graph capture failed for {k}: {type(exc).__name__})")
                    t[k + "|g"] = float("nan")
            best = min(t[s] for s in strategies if s != "merged")
            line = f"{tag:8s} {rank:5d} {t['base']*1000:11.1f} "
            for s in strategies:
                line += f"{(t[s]-t['base'])*1000:11.1f}"
            line += f"   {100*(best-t['base'])/t['base']:6.1f}%"
            print(line)
            gbase = t.get("base|g", float("nan"))
            gline = f"{'  graphed':8s} {rank:5d} {gbase*1000:11.1f} "
            for st in strategies:
                gline += f"{(t.get(st+'|g', float('nan'))-gbase)*1000:11.1f}"
            gbest = min(t.get(st + "|g", float("inf")) for st in strategies if st != "merged")
            gline += f"   {100*(gbest-gbase)/gbase:6.1f}%"
            print(gline)
            rows.append({"regime": label, "variant": tag, "rank": rank, "m": m,
                         "base_us": t["base"] * 1000,
                         **{s: t[s] * 1000 for s in strategies},
                         **{s + "_graphed": t.get(s + "|g", float("nan")) * 1000 for s in strategies},
                         "base_graphed_us": t.get("base|g", float("nan")) * 1000,
                         "best_sidecar_pct": 100 * (best - t["base"]) / t["base"]})
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")

    print("=== what compressing rank 64 -> 19 (e95) buys on a whole layer ===")
    for label, _ in (("decode  M=1", 1), ("prefill M=280", 280)):
        hi = next(r for r in rows if r["regime"] == label and r["variant"] == "full")
        lo = next(r for r in rows if r["regime"] == label and r["variant"] == "e95")
        best_hi = hi["base_us"] * (1 + hi["best_sidecar_pct"] / 100)
        best_lo = lo["base_us"] * (1 + lo["best_sidecar_pct"] / 100)
        print(f"  {label:14s} sidecar {hi['best_sidecar_pct']:5.1f}% -> {lo['best_sidecar_pct']:5.1f}%"
              f"   layer throughput {100*(best_hi/best_lo-1):+5.1f}%"
              f"   (naive: {hi['torch_naive']/hi['base_us']*100-100:5.1f}% -> "
              f"{lo['torch_naive']/lo['base_us']*100-100:5.1f}%)")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
