#!/usr/bin/env python3
"""Layer-configuration probe for the pool's outlier adapter.

dbpedia carries 99.88% of its update energy in 4 of 64 modules and S=25.3, so it
is the adapter every fleet rule ends up protecting and the one whose damage the
anchored model predicts worst: at k=187 the dense rule loses 0.008% of the update
energy and still drops to u=0.117, while the plain threshold rule at k=173 -- less
rank and 9x more spectral loss -- reaches 0.315.

Two things are separated here. `layer` holds the adapter budget at 187 and varies
only how those directions are spread, including rules the alpha family cannot
express (equalise the per-layer relative loss, protect the energy-carrying
modules, cap every module). `budget` holds the layer rule at the incumbent
energy greedy and varies K, to test whether u(k) is monotone at all -- the fleet
allocator's damage model assumes it is.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def energy_greedy(sig, K, cap=None, forced=None):
    L = len(sig)
    ranks = [1] * L
    if forced:
        for m, v in forced.items():
            ranks[m] = min(v, len(sig[m]))
    pool = sorted(((float(s[j] ** 2), m) for m, s in enumerate(sig)
                   for j in range(1, len(s))), reverse=True)
    left = K - sum(ranks)
    for _, m in pool:
        if left <= 0:
            break
        top = cap or len(sig[m])
        if ranks[m] < min(top, len(sig[m])):
            ranks[m] += 1
            left -= 1
    return ranks


def equal_loss(sig, K):
    """The threshold rule at an arbitrary K: every layer keeps the same fraction
    of its own energy. This is what the tau grid does, and no member of the
    sigma^2/tot^alpha family reproduces it."""
    cum = [np.cumsum(s ** 2) / float((s ** 2).sum()) for s in sig]
    def ranks_at(t):
        return [int(np.searchsorted(c, t) + 1) for c in cum]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(ranks_at(mid)) > K:
            hi = mid
        else:
            lo = mid
    r = ranks_at(lo)
    pool = sorted(((float(s[j] ** 2), m) for m, s in enumerate(sig)
                   for j in range(r[m], len(s))), reverse=True)
    left = K - sum(r)
    for _, m in pool:
        if left <= 0:
            break
        if r[m] < len(sig[m]):
            r[m] += 1
            left -= 1
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", type=Path, required=True)
    ap.add_argument("--adapter", default="dbpedia")
    ap.add_argument("--budget", type=int, default=187)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.spectra.read_text())[args.adapter]
    sig = [np.asarray(s, dtype=float) for s in spec["sigma"]]
    tot = np.array([float((s ** 2).sum()) for s in sig])
    top4 = list(np.argsort(tot)[-4:])
    K = args.budget

    configs = {
        "equalL":   equal_loss(sig, K),
        "am1":      None, "am0.5": None,           # filled below
        "a0.25":    None, "a0.75": None,
        "top4full": energy_greedy(sig, K, forced={int(m): len(sig[m]) for m in top4}),
        "cap4":     energy_greedy(sig, K, cap=4),
        "cap2":     energy_greedy(sig, K, cap=2),
    }
    for tag, alpha in (("am1", -1.0), ("am0.5", -0.5), ("a0.25", 0.25), ("a0.75", 0.75)):
        w = tot ** (-alpha)
        ranks = [1] * len(sig)
        pool = sorted(((float(s[j] ** 2) * w[m], m) for m, s in enumerate(sig)
                       for j in range(1, len(s))), reverse=True)
        for _, m in pool[:max(0, K - len(sig))]:
            ranks[m] += 1
        configs[tag] = ranks

    args.outdir.mkdir(parents=True, exist_ok=True)
    for tag, ranks in configs.items():
        if sum(ranks) != K:
            print(f"  {tag}: spends {sum(ranks)}, not {K} -- skipped"); continue
        kept = sum(float((sig[m][:ranks[m]] ** 2).sum()) for m in range(len(sig)))
        L = math.sqrt(max(0.0, 1.0 - kept / tot.sum()))
        (args.outdir / f"cfg_{tag}.json").write_text(json.dumps(
            {"tag": tag, "budget": K, "allocation": {
                args.adapter: {"k": K, "module_ranks": ranks, "L_W": L}}}, indent=2) + "\n")
        print(f"  {tag:10s} L_W={L:.4f}  rank1={sum(1 for x in ranks if x == 1):3d}  "
              f"top4={[ranks[int(m)] for m in top4]}")
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
