#!/usr/bin/env python3
"""Compare decode sidecar geometries on Qwen-shaped linear layers."""

import json
import statistics
from pathlib import Path

import torch

from loraforge_kernels import fused_lora, tiled_lora


def time_us(fn, warmup=10, repeats=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize()
        values.append(a.elapsed_time(b) * 1000)
    return statistics.median(values)


def main():
    torch.manual_seed(0)
    rows = []
    # Qwen language MLP/attention shapes; rank sweep covers aligned and exact E90/E95/E99.
    for m in (1, 4, 8, 32, 64):
        for k, n in ((2560, 9728), (9728, 2560), (2560, 4096), (2560, 1024)):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
            for r in (8, 13, 16, 21, 24, 27, 32):
                a = torch.randn(r, k, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(n, r, device="cuda", dtype=torch.bfloat16)

                def direct():
                    return y + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)

                reference = direct()
                variants = {
                    "direct": direct,
                    "row": lambda: fused_lora(x, a, b, y),
                    "tiled": lambda: tiled_lora(x, a, b, y),
                }
                timings = {}
                errors = {}
                for name, fn in variants.items():
                    got = fn()
                    errors[name] = float((got.float() - reference.float()).abs().max())
                    timings[name] = time_us(fn)
                winner = min(timings, key=timings.get)
                item = {"m": m, "k": k, "n": n, "r": r,
                        "timings_us": timings, "max_abs_error": errors, "winner": winner}
                rows.append(item)
                print(item, flush=True)
    out = Path("artifacts/kernel_bench/decode-tiles.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
