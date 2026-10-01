#!/usr/bin/env python3
"""Are the allocator's decisions portable when its coefficients are not?

Table 5 fits the law on four populations, two modalities and two measurement
locations, and the coefficients are plainly not portable: `a` spans 4.5x. Taken
alone that reads as an argument against using a fitted law to size adapters at
all, which would undercut Sec. 4.7.

But the allocator does not consume `a`, `b` and `c` separately. Its cap is

    log L_i <= (level - c)/b - (a/b) log S_i,

and the outer bisection solves for `level` at a fixed budget, so any shift in `c`
and any rescaling of `b` is absorbed. **The allocation depends on a/b alone.**
Across Table 5 that ratio spans 1.7x, not 4.5x.

This script asks what that residual spread does to the decisions themselves: it
runs the same minimax allocation at the same budget under every row's exponents
and compares the resulting per-adapter direction counts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predibase_budget_allocation import allocate, tabulate  # noqa: E402

# (label, a, b) for every row of Table 5, in table order
ROWS = [
    ("Lots-of-LoRAs $D_{prompt}$", 2.15, 2.84),
    ("Lots-of-LoRAs $D_{out}$",    1.45, 2.79),
    ("LoRA Land $D_{prompt}$",     1.38, 2.68),
    ("LoRA Land $D_{out}$",        1.59, 2.64),
    ("LoRARetriever $D_{prompt}$", 3.30, 3.77),
    ("LoRARetriever $D_{out}$",    2.74, 4.10),
    ("Wan2.1 video",               0.73, 1.01),
    ("Wan within-trajectory",      1.24, 1.42),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", type=Path, required=True,
                    help="json of {name: {S, sigma: [[...], ...]}} per adapter")
    ap.add_argument("--budgets", type=int, nargs="*", default=[])
    ap.add_argument("--tau-grid", type=float, nargs="+", default=[0.90, 0.80, 0.70, 0.50],
                    help="budgets are what uniform tau retains over the pool, which is "
                         "how Sec. 4.7 defines them; absolute counts differ by pool so "
                         "they cannot be carried across")
    ap.add_argument("--label", default="")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.spectra.read_text())
    adapters = {}
    for name, d in spec.items():
        sig = [torch.tensor(s, dtype=torch.float64) for s in d["sigma"]]

        def make(sig=sig):
            def at(tau):
                k_tot = 0
                keep = drop = 0.0
                for sv in sig:
                    e = sv.square()
                    c = torch.cumsum(e, 0)
                    k = min(int(torch.searchsorted(c, tau * c[-1]).item()) + 1, e.numel())
                    k_tot += k
                    keep += float(e[:k].sum())
                    drop += float(e[k:].sum())
                L = math.sqrt(drop / (keep + drop)) if keep + drop > 0 else 0.0
                return k_tot, L
            return at
        adapters[name] = {"S": d["S"], "at": make()}
    tabulate(adapters)
    names = sorted(adapters)
    budgets = args.budgets or [
        int(sum(ad["at"](t)[0] for ad in adapters.values())) for t in args.tau_grid]
    nominal = sum(len(d["sigma"]) * max(len(x) for x in d["sigma"]) for d in spec.values())
    print(f"{args.label or 'pool'}: {len(names)} adapters, {nominal} nominal directions")
    print(f"budgets from uniform tau {args.tau_grid}: {budgets}\n")

    out = {}
    for budget in budgets:
        allocs = {}
        for label, a, b in ROWS:
            al = allocate(adapters, budget, (a, b, 0.0))
            allocs[label] = np.array([al[n]["k"] for n in names], dtype=float)
        M = np.array([allocs[l] for l, _, _ in ROWS])
        ref = M[0]
        wide = len(names) > 10
        print(f"=== budget {budget} ===")
        head = f"{'row':30s}{'a/b':>7}{'corr':>8}{'max|dk|':>9}{'spearman':>10}"
        print(head if wide else head + "   " + " ".join(f"{n[:8]:>9}" for n in names))
        rank = lambda v: np.argsort(np.argsort(v))
        for (label, a, b), v in zip(ROWS, M):
            d = np.abs(v - ref).max()
            r = np.corrcoef(v, ref)[0, 1] if v.std() > 0 and ref.std() > 0 else 1.0
            rs = (np.corrcoef(rank(v), rank(ref))[0, 1]
                  if v.std() > 0 and ref.std() > 0 else 1.0)
            line = f"{label:30s}{a/b:7.3f}{r:8.3f}{d:9.0f}{rs:10.3f}"
            print(line if wide else line + "   " + " ".join(f"{x:9.0f}" for x in v))
        rel = np.abs(M - ref).max(axis=0) / np.maximum(ref, 1)
        print(f"  worst per-adapter disagreement across all eight rows: "
              f"{rel.max()*100:.1f}% of the reference allocation")
        pair = max(np.abs(M[i] - M[j]).max() for i in range(len(M)) for j in range(len(M)))
        print(f"  largest pairwise gap in directions: {pair:.0f}\n")
        out[budget] = {l: v.tolist() for l, v in zip([r[0] for r in ROWS], M)}
    out["adapters"] = names
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
