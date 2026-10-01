#!/usr/bin/env python3
"""Capacity benefit: how many adapters fit, and what it costs when they do not.

This is a different benefit from the kernel work.  Compression makes each
adapter smaller, so more of them stay resident in a fixed GPU adapter cache; the
payoff is not faster GEMMs but avoided host-to-device reloads.  It only appears
once the workload's adapter working set exceeds the cache, which is why the
interesting axis is the number of distinct adapters.

Sizes come from the measured FraQ AnyFlow adapters (rank 256 original, mean
retained rank 172/144/110/55 at e95/e90/e80/e50), scaled down by a constant so
the host can pin them; the cache budget is scaled by the same constant, so the
cliff sits at the same adapter count as it would at full size.

Reported per configuration:
  hit rate and bytes moved   -- implementation independent, the real result
  requests/s                 -- with the stated per-request compute, secondary
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

# Measured from artifacts/bghira/.../fraq_aligned8 plus the rank-256 original.
VARIANTS = [("original", 256, 637.0), ("e95", 172, 429.0), ("e90", 144, 345.0),
            ("e80", 110, 267.0), ("e50", 55, 115.0)]
SCALE = 32          # shrink adapters and the cache budget by the same factor
CACHE_ORIGINALS = 32  # budget = this many *original* adapters
COUNTS = [16, 32, 64, 128, 256]
REQUESTS = 600


class AdapterCache:
    """LRU over a fixed byte budget, holding real GPU buffers."""

    def __init__(self, budget_bytes):
        self.budget = budget_bytes
        self.used = 0
        self.slots = collections.OrderedDict()
        self.hits = self.misses = 0
        self.bytes_in = 0

    def get(self, key, staging):
        if key in self.slots:
            self.slots.move_to_end(key)
            self.hits += 1
            return self.slots[key]
        self.misses += 1
        need = staging.numel() * staging.element_size()
        while self.used + need > self.budget and self.slots:
            _, victim = self.slots.popitem(last=False)
            self.used -= victim.numel() * victim.element_size()
            del victim
        buf = torch.empty_like(staging, device="cuda")
        buf.copy_(staging, non_blocking=False)
        self.bytes_in += need
        self.slots[key] = buf
        self.used += need
        return buf


def workload(n_adapters, requests, dist, rng):
    if dist == "uniform":
        return rng.integers(0, n_adapters, requests)
    alpha = float(dist.replace("zipf", ""))
    w = 1.0 / np.arange(1, n_adapters + 1) ** alpha
    return rng.choice(n_adapters, size=requests, p=w / w.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/adapter_cache.json"))
    parser.add_argument("--requests", type=int, default=REQUESTS)
    args = parser.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(0)
    budget = int(CACHE_ORIGINALS * VARIANTS[0][2] * 2**20 / SCALE)

    # Host-to-device bandwidth decides how much a miss actually costs here.
    probe = torch.empty(int(64 * 2**20 // 2), dtype=torch.float16, pin_memory=True)
    dst = torch.empty_like(probe, device=dev)
    for _ in range(3):
        dst.copy_(probe)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        dst.copy_(probe)
    torch.cuda.synchronize()
    h2d = 10 * probe.numel() * 2 / (time.perf_counter() - t0) / 1e9
    del probe, dst
    print(f"GPU: {torch.cuda.get_device_name(0)}   H2D {h2d:.0f} GB/s   "
          f"cache budget {budget/2**20:.0f} MiB (= {CACHE_ORIGINALS} original adapters)")
    print(f"adapter sizes scaled 1/{SCALE}: " +
          ", ".join(f"{n}={s/SCALE:.1f}MiB" for n, _, s in VARIANTS))

    # The first configuration measured pays allocator and context warm-up, which
    # showed up as a 5x outlier in the first cell; spend one throwaway run first.
    warm_stage = [torch.empty(1 << 20, dtype=torch.float16, pin_memory=True) for _ in range(8)]
    warm_cache = AdapterCache(budget)
    for a in workload(8, 64, "uniform", rng):
        warm_cache.get(int(a), warm_stage[int(a)])[:1024].mul_(1.0)
    torch.cuda.synchronize()
    del warm_stage, warm_cache
    torch.cuda.empty_cache()

    rows = []
    for dist in ("uniform", "zipf1.0"):
        print(f"\n=== traffic: {dist} ===")
        print(f"{'variant':9s} {'rank':>5s} {'MiB':>6s} {'fits':>5s} | " +
              " ".join(f"{'K='+str(k):>21s}" for k in COUNTS))
        for name, rank, mib in VARIANTS:
            size = int(mib * 2**20 / SCALE)
            fits = budget // size
            line = f"{name:9s} {rank:5d} {size/2**20:6.1f} {fits:5d} | "
            for k in COUNTS:
                staging = [torch.empty(size // 2, dtype=torch.float16, pin_memory=True)
                           for _ in range(k)]
                cache = AdapterCache(budget)
                order = workload(k, args.requests, dist, rng)
                lat = []
                torch.cuda.synchronize()
                started = time.perf_counter()
                for a in order:
                    t0 = time.perf_counter()
                    buf = cache.get(int(a), staging[int(a)])
                    # Touch the adapter so a hit is not free either.
                    buf[:1024].mul_(1.0)
                    torch.cuda.synchronize()
                    lat.append((time.perf_counter() - t0) * 1e3)
                elapsed = time.perf_counter() - started
                hit = 100 * cache.hits / (cache.hits + cache.misses)
                rps = args.requests / elapsed
                p99 = sorted(lat)[int(0.99 * len(lat))]
                line += f"{hit:4.0f}% {rps:6.0f}/s {p99:5.2f}ms "
                rows.append({"dist": dist, "variant": name, "rank": rank,
                             "adapter_mib": size / 2**20, "resident_capacity": int(fits),
                             "adapters": k, "hit_rate_pct": hit, "requests_per_s": rps,
                             "gib_transferred": cache.bytes_in / 2**30,
                             "p50_ms": statistics.median(lat),
                             "p99_ms": p99,
                             "h2d_gbps": h2d})
                del staging, cache
                torch.cuda.empty_cache()
            print(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0),
                                       "budget_mib": budget / 2**20, "scale": SCALE,
                                       "rows": rows}, indent=2) + "\n")
    print("\n=== resident capacity at a fixed budget ===")
    for name, rank, mib in VARIANTS:
        print(f"  {name:9s} rank {rank:3d}  {mib:5.0f} MiB each  -> "
              f"{int(budget // (mib * 2**20 / SCALE)):3d} adapters resident "
              f"({budget // (mib * 2**20 / SCALE) / (budget // (VARIANTS[0][2] * 2**20 / SCALE)):.2f}x)")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
