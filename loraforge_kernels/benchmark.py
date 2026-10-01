#!/usr/bin/env python3
"""Microbenchmark the four LoRA execution-path optimization prototypes."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from loraforge_kernels import fused_lora, grouped_lora


def time_us(fn, warmup=40, repeats=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record(); fn(); end.record(); end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000)
    return {"median_us": statistics.median(samples), "mean_us": statistics.fmean(samples),
            "std_us": statistics.stdev(samples)}


def main():
    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    k = n = 4096
    rows = []
    for m in (1, 4, 8):
        for r in (1, 4, 8, 16, 32):
            x = torch.randn(m, k, device=device, dtype=dtype)
            a = torch.randn(r, k, device=device, dtype=dtype) / k**0.5
            b = torch.randn(n, r, device=device, dtype=dtype) / max(r, 1)**0.5
            y = torch.randn(m, n, device=device, dtype=dtype)
            ref = y + (x @ a.T) @ b.T
            got = fused_lora(x, a, b, y)
            max_abs = (ref - got).abs().max().item()
            baseline = time_us(lambda: y + (x @ a.T) @ b.T)
            no_add = time_us(lambda: (x @ a.T) @ b.T)
            fused = time_us(lambda: fused_lora(x, a, b, y))
            rows.append({"kind":"single", "m":m, "r":r, "max_abs":max_abs,
                         "baseline":baseline, "two_gemm_no_add":no_add, "fused":fused,
                         "speedup":baseline["median_us"]/fused["median_us"]})
            print(json.dumps(rows[-1]), flush=True)

    for p, label in ((2, "gate_up"), (3, "qkv")):
        for m in (1, 4, 8):
            for r in (4, 8, 16, 32):
                x = torch.randn(m, k, device=device, dtype=dtype)
                aa = torch.randn(p, r, k, device=device, dtype=dtype) / k**0.5
                a = aa.reshape(p*r, k).contiguous()
                b = torch.randn(p, n, r, device=device, dtype=dtype) / r**0.5
                y = torch.randn(m, p, n, device=device, dtype=dtype)
                def naive():
                    return torch.stack([y[:,i] + (x @ aa[i].T) @ b[i].T for i in range(p)], 1)
                def concat_a():
                    z = (x @ a.T).reshape(m,p,r)
                    return torch.stack([y[:,i] + z[:,i] @ b[i].T for i in range(p)], 1)
                ref = naive()
                got = grouped_lora(x, a, b, y)
                max_abs = (ref-got).abs().max().item()
                naive_t, concat_t = time_us(naive), time_us(concat_a)
                grouped_t = time_us(lambda: grouped_lora(x,a,b,y))
                rows.append({"kind":label,"p":p,"m":m,"r":r,"max_abs":max_abs,
                             "naive":naive_t,"concat_a":concat_t,"grouped":grouped_t,
                             "speedup_vs_naive":naive_t["median_us"]/grouped_t["median_us"]})
                print(json.dumps(rows[-1]), flush=True)
    out = Path("artifacts/kernel_bench/four_cuts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu":torch.cuda.get_device_name(),"rows":rows},indent=2)+"\n")


if __name__ == "__main__": main()
