#!/usr/bin/env python3
"""Fast sweep of sidecar throughput techniques on a Wan-like transformer block.

Layer-at-a-time microbenchmarks miss the technique that matters most -- the
projections in a block share inputs, so work can be amortised across them -- and
end-to-end runs are far too slow to iterate on.  This times the block's linear
stack under each strategy in one short job.

  merged      : W += scale*B@A, one GEMM per layer.  Not swappable; the floor.
  torch       : the naive three-pass sidecar.
  hybrid      : cuBLAS GEMMs + fused expand-add epilogue.
  concat      : [W ; A] packed along N, one GEMM, fused epilogue.
  kconcat     : [x | scale*z] built per layer, one wide-K GEMM.
  group_kconcat : [x | z_1 | .. | z_P] built ONCE per group of layers sharing an
                  input, each layer then a wide-K GEMM.  Amortises the only
                  rank-independent term kconcat has.
  group_hybrid  : one shrink GEMM for the whole group, then per layer a base
                  GEMM and the fused expand-add epilogue.
  group_concat  : the group's leader packs [W_0 ; A_1..A_P], so its own GEMM
                  emits every z in one pass over x -- no separate shrink at all.
  cublas_accum  : hybrid with the expand-add done by cuBLAS addmm(beta=1)
                  instead of the Triton epilogue, to check the epilogue is
                  actually worth having.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from loraforge_kernels.fused_linear import (
    _concat, _hybrid, _kconcat, _torch_naive,
    augment_bias, augment_weight, expand_add, kconcat_weight, packed_rank,
)

# (group name, in_features, [out_features...]) -- layers in one group share x.
WAN_BLOCK = [
    ("self_qkv", 1536, [1536, 1536, 1536]),
    ("self_o", 1536, [1536]),
    ("cross_q", 1536, [1536]),
    ("cross_kv", 1536, [1536, 1536]),
    ("ffn_up", 1536, [8960]),
    ("ffn_down", 8960, [1536]),
]


class Group:
    def __init__(self, name, k, outs, rank, m, dtype, dev):
        self.name, self.k, self.rank = name, k, rank
        self.x = torch.randn(m, k, device=dev, dtype=dtype) / k**0.5
        self.w = [torch.randn(n, k, device=dev, dtype=dtype) / k**0.5 for n in outs]
        self.bias = [torch.randn(n, device=dev, dtype=dtype) for n in outs]
        self.a = [torch.randn(rank, k, device=dev, dtype=dtype) / k**0.5 for _ in outs]
        self.b = [torch.randn(n, rank, device=dev, dtype=dtype) / rank**0.5 for n in outs]
        self.scale = 1.0
        self.outs = outs
        # Pre-packed forms; building these is a load-time cost, not per-call.
        self.w_merged = [w + self.scale * (b @ a) for w, a, b in zip(self.w, self.a, self.b)]
        self.w_aug = [augment_weight(w, a) for w, a in zip(self.w, self.a)]
        self.b_aug = [augment_bias(bi, rank) for bi in self.bias]
        self.w_kc = [kconcat_weight(w, b) for w, b in zip(self.w, self.b)]
        self.rank_pad = packed_rank(rank)
        # Group form: one A stacked over the whole group, and each layer's B
        # padded with zeros into the group's slot so one ext serves them all.
        p = len(outs)
        self.a_cat = torch.cat(self.a, dim=0).contiguous()
        self.group_pad = packed_rank(p * rank)
        # Leader packing: layer 0's weight carries the whole group's A rows, so
        # its GEMM emits [y_0 | z_1..z_P] in one pass over x.
        self.w_aug_leader = torch.cat([self.w[0], self.a_cat], dim=0).contiguous()
        self.b_aug_leader = (torch.cat([self.bias[0], self.bias[0].new_zeros(p * rank)])
                             if self.bias[0] is not None else None)
        self.w_kc_group = []
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            packed = w.new_zeros((w.shape[0], k + self.group_pad))
            packed[:, :k] = w
            packed[:, k + i * rank : k + (i + 1) * rank] = b
            self.w_kc_group.append(packed.contiguous())

    def base(self):
        return [torch.nn.functional.linear(self.x, w, bi) for w, bi in zip(self.w, self.bias)]

    def merged(self):
        return [torch.nn.functional.linear(self.x, w, bi) for w, bi in zip(self.w_merged, self.bias)]

    def torch_naive(self):
        return [_torch_naive(self.x, w, bi, a, b, self.scale)
                for w, bi, a, b in zip(self.w, self.bias, self.a, self.b)]

    def hybrid(self):
        return [_hybrid(self.x, w, bi, a, b, self.scale)
                for w, bi, a, b in zip(self.w, self.bias, self.a, self.b)]

    def concat(self):
        return [_concat(self.x, wa, ba, b, self.scale, n)
                for wa, ba, b, n in zip(self.w_aug, self.b_aug, self.b, self.outs)]

    def kconcat(self):
        return [_kconcat(self.x, wk, bi, a, self.scale, self.rank_pad)
                for wk, bi, a in zip(self.w_kc, self.bias, self.a)]

    def group_kconcat(self):
        m, k = self.x.shape
        total = self.a_cat.shape[0]
        ext = torch.empty((m, k + self.group_pad), device=self.x.device, dtype=self.x.dtype)
        ext[:, :k].copy_(self.x)
        ext[:, k : k + total].copy_(torch.mm(self.x, self.a_cat.t()).mul_(self.scale))
        if self.group_pad > total:
            ext[:, k + total :].zero_()
        return [torch.nn.functional.linear(ext, wk, bi)
                for wk, bi in zip(self.w_kc_group, self.bias)]


    def group_hybrid(self):
        z = torch.mm(self.x, self.a_cat.t())
        out = []
        for i, (w, bi, b) in enumerate(zip(self.w, self.bias, self.b)):
            y = torch.nn.functional.linear(self.x, w, bi)
            out.append(expand_add(y, z[:, i * self.rank : (i + 1) * self.rank], b,
                                  self.scale))
        return out

    def group_concat(self):
        n0 = self.outs[0]
        y_aug = torch.nn.functional.linear(self.x, self.w_aug_leader, self.b_aug_leader)
        z = y_aug[:, n0:]
        out = [expand_add(y_aug[:, :n0], z[:, : self.rank], self.b[0], self.scale)]
        for i in range(1, len(self.outs)):
            y = torch.nn.functional.linear(self.x, self.w[i], self.bias[i])
            out.append(expand_add(y, z[:, i * self.rank : (i + 1) * self.rank],
                                  self.b[i], self.scale))
        return out

    def cublas_accum(self):
        out = []
        for w, bi, a, b in zip(self.w, self.bias, self.a, self.b):
            y = torch.nn.functional.linear(self.x, w, bi)
            z = torch.mm(self.x, a.t())
            out.append(torch.addmm(y, z, b.t(), beta=1.0, alpha=self.scale))
        return out


STRATEGIES = ("merged", "torch_naive", "hybrid", "cublas_accum", "concat",
              "group_concat", "group_hybrid", "kconcat", "group_kconcat")


def time_all(fns, warmup=5, rounds=12):
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    best = {k: float("inf") for k in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            b, e = torch.cuda.Event(True), torch.cuda.Event(True)
            b.record(); fn(); e.record(); e.synchronize()
            best[name] = min(best[name], b.elapsed_time(e))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32760)
    parser.add_argument("--ranks", type=int, nargs="+", default=[256, 144, 56])
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/techniques.json"))
    args = parser.parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  M={args.m}")
    rows = []

    for rank in args.ranks:
        groups = [Group(name, k, outs, rank, args.m, dtype, dev) for name, k, outs in WAN_BLOCK]
        # Accuracy guard: every strategy must agree with the naive one.
        for g in groups:
            want = g.torch_naive()
            for s in STRATEGIES:
                got = getattr(g, s)()
                err = max(float((x.float() - y.float()).abs().max() / y.abs().max().clamp_min(1e-6))
                          for x, y in zip(got, want))
                if err > 5e-2:
                    print(f"  !! {g.name}/{s} rel_err={err:.2e}")

        per_group = {}
        for g in groups:
            fns = {"base": g.base}
            fns.update({s: getattr(g, s) for s in STRATEGIES})
            per_group[g.name] = time_all(fns)

        print(f"\n=== rank {rank}: sidecar us over the un-adapted block, per group ===")
        header = f"{'group':12s} {'base us':>9s} " + " ".join(f"{s[:10]:>11s}" for s in STRATEGIES)
        print(header)
        totals = {s: 0.0 for s in STRATEGIES}
        base_total = 0.0
        for g in groups:
            t = per_group[g.name]
            base_total += t["base"]
            line = f"{g.name:12s} {t['base']*1000:9.1f} "
            for s in STRATEGIES:
                totals[s] += t[s]
                line += f"{(t[s]-t['base'])*1000:11.1f}"
            print(line)
            rows.append({"rank": rank, "group": g.name, "base_us": t["base"] * 1000,
                         **{s: t[s] * 1000 for s in STRATEGIES}})

        print(f"\n  block total: base {base_total*1000:.0f} us")
        best_mix = 0.0
        for s in STRATEGIES:
            print(f"    {s:14s} {totals[s]*1000:9.0f} us   sidecar {100*(totals[s]-base_total)/base_total:6.1f}%")
        # Per group, pick the best swappable strategy: that is what the runtime
        # selector would do, so it is the number that predicts end-to-end.
        for g in groups:
            t = per_group[g.name]
            best_mix += min(t[s] for s in STRATEGIES if s != "merged")
        print(f"    {'per-group best':14s} {best_mix*1000:9.0f} us   "
              f"sidecar {100*(best_mix-base_total)/base_total:6.1f}%")
        print(f"    {'merged (floor)':14s} {totals['merged']*1000:9.0f} us   "
              f"sidecar {100*(totals['merged']-base_total)/base_total:6.1f}%")
        rows.append({"rank": rank, "group": "__block__", "base_us": base_total * 1000,
                     "per_group_best_us": best_mix * 1000,
                     **{s: totals[s] * 1000 for s in STRATEGIES}})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
