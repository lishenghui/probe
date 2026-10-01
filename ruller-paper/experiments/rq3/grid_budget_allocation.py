#!/usr/bin/env python3
"""Uniform vs strength-calibrated allocation on task metrics, over a measured grid.

Sec. 4.7 answers the question that matters -- does allocating by strength protect
downstream utility -- on seven LoRA Land adapters, because that is the only pool
whose evaluation harness was wired to the allocator. Two other pools already have
their task metrics measured on a threshold grid, so the same comparison can be run
on 41 LoraRetriever and 19 Lots-of-LoRAs adapters without generating anything new.

The allocator is restricted to the thresholds that were actually evaluated. That
is not a compromise: it removes interpolation from the utility side entirely, so
every number reported here is a measurement rather than a fit through
measurements, and choosing among a handful of levels is what a deployment does
anyway.

Retention is scored against the un-adapted base model on the same prompts, as
everywhere else in the paper, so u < 0 means the compressed adapter is worse than
attaching no adapter at all.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

TAUS = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]


def load_task(patterns):
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for r in json.loads(Path(f).read_text()):
                name = r.get("short") or r.get("adapter")
                e = out.setdefault(name, {k: v for k, v in r.items() if k != "variants"})
                e.setdefault("variants", {}).update(r["variants"])
    return out


def load_div(patterns):
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for r in json.loads(Path(f).read_text()):
                name = r.get("short") or r.get("adapter")
                out.setdefault(name, {"S": r.get("S"), "variants": {}})["variants"].update(
                    r["variants"])
    return out


def fit(cells):
    X = np.array([[1.0, math.log(s), math.log(l)] for s, l, d in cells])
    y = np.array([math.log(d) for _, _, d in cells])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[0], b[1], b[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="+", required=True)
    ap.add_argument("--div", nargs="+", required=True)
    ap.add_argument("--anchor-div", nargs="+", default=None,
                    help="optional disjoint-prompt divergence files used only for "
                         "the per-adapter anchor")
    ap.add_argument("--nominal", type=int, required=True,
                    help="directions per adapter, r x modules")
    ap.add_argument("--min-headroom", type=float, default=0.05)
    ap.add_argument("--max-u-quantum", type=float, default=None,
                    help="drop adapters whose retained utility the evaluation set "
                         "cannot resolve. One prompt flipping moves u by "
                         "1/(n*headroom); where that quantum is large the whole "
                         "compression curve is a handful of prompts and the adapter "
                         "can take over the worst-case column on noise alone. "
                         "Headroom alone does not catch this: an adapter can clear "
                         "a 0.05 headroom bar on 30 prompts and still move 0.2 in u "
                         "per prompt.")
    ap.add_argument("--label", default="pool")
    ap.add_argument("--anchor-tau", type=float, default=None,
                    help="optionally absorb each adapter's risk intercept with one "
                         "measured D_out anchor; uses the LOAO shared L_W slope and "
                         "requires no downstream labels")
    ap.add_argument("--fixed-b", type=float, default=None,
                    help="external within-adapter L_W exponent; with an anchor this "
                         "avoids fitting any target-pool compression sweep")
    ap.add_argument("--estimate-b-from-anchors", type=float, nargs=2, default=None,
                    metavar=("TAU1", "TAU2"), help="estimate the shared slope from "
                         "two unlabeled anchor levels using within-adapter ratios")
    ap.add_argument("--secondary-objective", choices=("none", "mean-d"), default="none",
                    help="after minimizing the worst predicted damage, optionally "
                         "minimize mean predicted divergence without relaxing that ceiling")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    task, div = load_task(args.task), load_div(args.div)
    anchor_div = load_div(args.anchor_div) if args.anchor_div else div
    if args.estimate_b_from_anchors:
        t1,t2=args.estimate_b_from_anchors
        l1,l2=(f"e{round(t*100):02d}" for t in (t1,t2))
        xs,ys=[],[]
        for n,r in anchor_div.items():
            if l1 not in r["variants"] or l2 not in r["variants"]: continue
            v1,v2=r["variants"][l1],r["variants"][l2]
            if min(v1.get("d_out",0),v2.get("d_out",0),v1.get("L_W",0),v2.get("L_W",0))<=0: continue
            xs.append(math.log(v2["L_W"]/v1["L_W"]))
            ys.append(math.log(v2["d_out"]/v1["d_out"]))
        args.fixed_b=float(np.dot(xs,ys)/np.dot(xs,xs))
        print(f"two-anchor slope from {len(xs)} adapters: b={args.fixed_b:.6f}")
    names = sorted(n for n, r in task.items()
                   if (r.get("headroom") if "headroom" in r
                       else r["metric_orig"] - r["metric_base"]) >= args.min_headroom
                   and n in div and div[n].get("S")
                   and (args.max_u_quantum is None or r.get("n", 0) *
                        (r.get("headroom") if "headroom" in r
                         else r["metric_orig"] - r["metric_base"])
                        * args.max_u_quantum >= 1.0)
                   and all(f"e{round(t*100):02d}" in r["variants"] for t in TAUS))
    print(f"{args.label}: {len(names)} adapters with both a task metric and a strength")

    S = {n: div[n]["S"] for n in names}
    grid = {}
    for n in names:
        r = task[n]
        fl = r["metric_base"]
        hd = (r.get("headroom") if "headroom" in r else r["metric_orig"] - fl)
        rows = []
        for t in TAUS:
            v = r["variants"][f"e{round(t*100):02d}"]
            u = v.get("retained")
            if u is None:
                u = (v["metric"] - fl) / hd
            rows.append(dict(tau=t, k=int(round(v["rank_frac"] * args.nominal)),
                             L=v["L_W"], u=float(u)))
        grid[n] = rows

    # leave-one-adapter-out: no adapter is sized by a law its own divergence fitted
    cells = {n: [(div[n]["S"], v["L_W"], v["d_out"])
                 for v in div[n]["variants"].values()
                 if v.get("d_out", 0) > 0 and v.get("L_W", 0) > 0] for n in names}
    beta = {n: fit([c for m in names if m != n for c in cells[m]]) for n in names
            if sum(len(cells[m]) for m in names if m != n) > 3}

    def risk(n, row):
        c, a, b_fit = beta[n]
        b = args.fixed_b if args.fixed_b is not None else b_fit
        if args.anchor_tau is None:
            return c + a * math.log(S[n]) + b * math.log(max(row["L"], 1e-9))
        label = f"e{round(args.anchor_tau * 100):02d}"
        anchor = anchor_div[n]["variants"][label]
        if anchor.get("d_out", 0) <= 0 or anchor.get("L_W", 0) <= 0:
            raise ValueError(f"{n} has no positive D_out anchor at {label}")
        return (math.log(anchor["d_out"]) + b *
                (math.log(max(row["L"], 1e-9)) - math.log(anchor["L_W"])))

    def sct(budget):
        """Coarsest grid level per adapter whose predicted damage stays under a cap."""
        def alloc(level):
            out = {}
            for n in names:
                best = grid[n][0]
                for row in grid[n]:                      # tau descending, k descending
                    pred = risk(n, row)
                    if pred <= level:
                        best = row
                out[n] = best
            return out
        lo, hi = -60.0, 20.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if sum(v["k"] for v in alloc(mid).values()) > budget:
                lo = mid
            else:
                hi = mid
        if args.secondary_objective == "mean-d":
            # Lexicographic second stage.  Among all measured-rank allocations
            # that preserve the optimal minimax ceiling, minimize total
            # predicted divergence.  This uses the same unlabeled anchors as
            # the first stage and never consults downstream utility.
            feasible = {n: [r for r in grid[n] if risk(n, r) <= hi + 1e-9]
                        for n in names}
            inf = float("inf")
            dp = np.full(budget + 1, inf)
            dp[0] = 0.0
            parents = []
            for n in names:
                nd = np.full(budget + 1, inf)
                choice = np.full(budget + 1, -1, dtype=int)
                previous = np.full(budget + 1, -1, dtype=int)
                for j, row in enumerate(feasible[n]):
                    k = row["k"]
                    vals = dp[:budget + 1 - k] + math.exp(risk(n, row))
                    better = vals < nd[k:]
                    inds = np.flatnonzero(better) + k
                    nd[inds] = vals[better]
                    choice[inds] = j
                    previous[inds] = inds - k
                dp = nd
                parents.append((choice, previous))
            spent = int(np.argmin(dp))
            if not np.isfinite(dp[spent]):
                raise RuntimeError("no allocation satisfies the minimax ceiling")
            out = {}
            cursor = spent
            for i in range(len(names) - 1, -1, -1):
                choice, previous = parents[i]
                j = int(choice[cursor])
                out[names[i]] = feasible[names[i]][j]
                cursor = int(previous[cursor])
            return out
        a = alloc(hi)
        # trim any overspend from the largest allocation, the adapter SCT protects,
        # so the correction cannot flatter it
        while sum(v["k"] for v in a.values()) > budget:
            n = max(names, key=lambda m: a[m]["k"])
            nxt = [r for r in grid[n] if r["k"] < a[n]["k"]]
            if not nxt:
                break
            a[n] = max(nxt, key=lambda r: r["k"])
        # The common risk cap is discrete: its coarsest feasible grid choices can
        # leave capacity unused.  Spend that slack on upgrades, prioritising the
        # currently highest-risk adapter whose next measured level fits.  Every
        # move increases retained rank and therefore cannot worsen the minimax
        # surrogate; it also makes the capacity comparison as close to exact as
        # the measured rank grid permits.
        while True:
            spent = sum(v["k"] for v in a.values())
            candidates = []
            for n in names:
                upgrades = [r for r in grid[n]
                            if r["k"] > a[n]["k"] and r["k"] - a[n]["k"] <= budget - spent]
                if upgrades:
                    nxt = min(upgrades, key=lambda r: r["k"])
                    candidates.append((risk(n, a[n]), n, nxt))
            if not candidates:
                break
            _, n, nxt = max(candidates, key=lambda z: z[0])
            a[n] = nxt
        return a

    def stats(alloc):
        u = np.array([alloc[n]["u"] for n in names])
        return dict(mean=float(u.mean()), worst=float(u.min()),
                    p10=float(np.percentile(u, 10)), median=float(np.median(u)),
                    below09=int((u < 0.9).sum()), broken=int((u <= 0).sum()),
                    spent=int(sum(alloc[n]["k"] for n in names)))

    print(f"\n{'budget':>8} {'rule':8} {'mean':>7}{'worst':>8}{'p10':>8}{'median':>8}"
          f"{'u<0.9':>7}{'broken':>8}")
    out = {}
    for t in (0.90, 0.80, 0.70, 0.50):
        uni = {n: next(r for r in grid[n] if abs(r["tau"] - t) < 1e-9) for n in names}
        budget = sum(v["k"] for v in uni.values())
        chosen = sct(budget)
        s_uni, s_sct = stats(uni), stats(chosen)
        out[budget] = {"tau": t, "uniform": s_uni, "sct": s_sct,
                       "paired": {n: {"uniform": uni[n], "sct": chosen[n],
                                      "delta_u": chosen[n]["u"] - uni[n]["u"]}
                                  for n in names}}
        for lab, st in (("uniform", s_uni), ("SCT", s_sct)):
            print(f"{budget:8d} {lab:8} {st['mean']:7.3f}{st['worst']:8.3f}{st['p10']:8.3f}"
                  f"{st['median']:8.3f}{st['below09']:7d}{st['broken']:8d}")
        print()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"pool": args.label, "n": len(names),
                                       "anchor_tau": args.anchor_tau,
                                       "fixed_b": args.fixed_b,
                                       "budgets": out}, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
