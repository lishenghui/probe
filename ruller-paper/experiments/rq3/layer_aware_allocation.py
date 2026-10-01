#!/usr/bin/env python3
"""Layer-level rank allocation inside one adapter, at a fixed adapter budget.

The dense allocator solves the fleet problem -- how many directions each adapter
gets -- but inside an adapter it keeps the globally largest singular directions,
which is the exact minimum-Frobenius-loss choice and ignores the base weight the
update sits on. This is the two-level structure the fleet rule already assumes,
applied one level down:

    layer level    minimise  A_l [ S_l^gamma L_W,l(k_l) ]   s.t.  sum_l k_l <= K
    fleet level    minimise  max_i D_i(K_i)                 s.t.  sum_i K_i <= B

With A = sum of squares the layer objective is separable and the greedy is exact:

    sum_l S_l^{2g} L^2_{W,l} = sum_l drop_l * tot_l^{g-1} / ||W_l||^{2g}

so a direction's marginal cost is sigma^2 * tot_l^{g-1} / ||W_l||^{2g} and keeping
the largest ones is optimal at every K.  gamma=0 normalises each layer by its own
update energy; gamma=1 gives ||dropped_l||/||W_l||, the model-relative
perturbation; the incumbent rule is the gamma-free weight 1.

Any gamma other than the incumbent *raises* adapter-level L_W at the same K --
Frobenius optimality is exactly what is being traded away -- so this is only
worth running where the end-to-end divergence can be measured. K_i is copied
from the reference allocation so the fleet split is held fixed and the layer
question is isolated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def allocate(sigma, wnorm, K, alpha=0.0, beta=0.0, equal=False):
    """Greedy on sigma^2 * tot_l^-alpha * ||W_l||^-2beta, or an equal split.

    alpha discounts a layer by how much update energy it already carries and
    beta by the base weight the update sits on, so the two ingredients of
    S_l^gamma L_W,l can be tested apart:

        alpha=0,       beta=0      incumbent, exact minimum Frobenius loss
        alpha=1-gamma, beta=gamma  the S_l^gamma L_W,l family
        alpha=a,       beta=0      balance only, never reads a base norm
        alpha=0,       beta=b      base-norm weighting only

    `equal` ignores the spectrum entirely and is the control for "the gain is
    just from spreading rank across layers".
    """
    sig = [np.asarray(s, dtype=float) for s in sigma]
    w = np.asarray(wnorm, dtype=float)
    tot = np.array([float((s ** 2).sum()) for s in sig])
    L = len(sig)
    ranks = [1] * L                             # a rank-0 module is not a LoRA arm
    if equal:
        # spread the budget as evenly as the per-module rank ceilings allow;
        # the remainder goes to the lowest module indices, which is arbitrary but
        # fixed and touches at most one direction per layer
        left = max(0, K - L)
        while left > 0:
            room = [m for m in range(L) if ranks[m] < len(sig[m])]
            if not room:
                break
            lo = min(ranks[m] for m in room)
            step = [m for m in room if ranks[m] == lo]
            for m in step[:left]:
                ranks[m] += 1
            left -= min(len(step), left)
    else:
        weight = tot ** (-alpha) / w ** (2.0 * beta)
        pool = sorted(((float(s[j] ** 2) * weight[m], m)
                       for m, s in enumerate(sig) for j in range(1, len(s))), reverse=True)
        for _, m in pool[:max(0, K - L)]:
            ranks[m] += 1
    kept = sum(float((sig[m][:ranks[m]] ** 2).sum()) for m in range(L))
    return ranks, math.sqrt(max(0.0, 1.0 - kept / tot.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", type=Path, required=True,
                    help="must carry per-module wnorm, not only sigma")
    ap.add_argument("--reference", type=Path, required=True,
                    help="dense allocation whose per-adapter K is held fixed")
    ap.add_argument("--gamma", type=float, default=None,
                    help="shorthand for --alpha 1-gamma --beta gamma")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--equal", action="store_true")
    ap.add_argument("--tag", default=None, help="label recorded in the output")
    ap.add_argument("--only-movable", action="store_true",
                    help="drop adapters sitting at the structural floor, where "
                         "K equals the module count and no layer choice exists")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    alpha, beta = args.alpha, args.beta
    if args.gamma is not None:
        alpha, beta = 1.0 - args.gamma, args.gamma
    tag = args.tag or ("equal" if args.equal else f"a{alpha:g}_b{beta:g}")

    spec = json.loads(args.spectra.read_text())
    ref = json.loads(args.reference.read_text())
    out, skipped = {}, []
    for name, row in ref["allocation"].items():
        if name not in spec:
            skipped.append((name, "no spectra")); continue
        s = spec[name]
        if "wnorm" not in s:
            raise SystemExit(f"{name}: spectra file has no per-module wnorm")
        K = int(row["k"])
        if args.only_movable and K <= len(s["sigma"]):
            skipped.append((name, "at the floor")); continue
        ranks, L = allocate(s["sigma"], s["wnorm"], K, alpha, beta, args.equal)
        base, _ = allocate(s["sigma"], s["wnorm"], K)
        moved = int(sum(abs(a - b) for a, b in zip(ranks, base)) // 2)
        out[name] = {"k": K, "module_ranks": ranks, "L_W": L,
                     "L_W_energy_rule": row.get("L_W"), "moved": moved}
        print(f"  {name:16s} K={K:5d}  L_W {row.get('L_W', float('nan')):.4f}"
              f" -> {L:.4f}   {moved:3d} directions moved", flush=True)
    for name, why in skipped:
        print(f"  {name:16s} skipped: {why}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"pool": ref.get("pool"), "rule": tag, "alpha": alpha, "beta": beta,
         "equal": args.equal, "gamma": args.gamma, "budget": ref.get("budget"),
         "reference": str(args.reference), "n": len(out), "allocation": out},
        indent=2) + "\n")
    print(f"wrote {args.output} ({len(out)} adapters)")


if __name__ == "__main__":
    main()
