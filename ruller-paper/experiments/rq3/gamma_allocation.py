#!/usr/bin/env python3
"""One-parameter family of compression policies, scored on measured task utility.

The allocator's cap can be written log L_i <= l - gamma log S_i, that is

    S_i^gamma L_i(k_i) ~ const,

with l set by bisection to meet a fixed total rank budget. Every rule in this
paper is a member of that family:

    gamma = 0     equal L_W for every adapter, which is uniform tau
    gamma = 1     equal S L_W, the model-relative perturbation, needing no fit
    gamma = a/b   the calibrated rule, needing a fitted law

Presenting them as one family rather than as method-versus-baseline makes the
question a practitioner actually faces -- how much should strength tilt the
allocation -- and lets the calibration-free setting be read off directly. If
gamma = 1 recovers most of what the fitted gamma buys, a fleet operator needs one
scalar per adapter and no divergence measurements at all.

Levels are restricted to the thresholds and ranks whose task utility was
measured, so every reported utility is an observation.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

TAUS = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]


def load(patterns):
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for r in json.loads(Path(f).read_text()):
                n = r.get("short") or r.get("adapter")
                e = out.setdefault(n, {})
                e.update({k: v for k, v in r.items() if k != "variants"})
                e.setdefault("variants", {}).update(r["variants"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="+", required=True)
    ap.add_argument("--strengths", type=Path, required=True)
    ap.add_argument("--nominal", type=int, required=True)
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    ap.add_argument("--named", nargs="*", default=[],
                    help="extra gammas as name=value, e.g. 'foreign a/b=0.520'")
    ap.add_argument("--min-headroom", type=float, default=0.05)
    ap.add_argument("--label", default="pool")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    task = load(args.task)
    spec = json.loads(args.strengths.read_text())
    S = ({e["adapter"]: e.get("S_global", e.get("S")) for e in spec}
         if isinstance(spec, list) and "adapter" in spec[0]
         else {e["task"]: e["S"] for e in spec} if isinstance(spec, list)
         else {k: v["S"] for k, v in spec.items()})
    names = sorted(n for n, r in task.items()
                   if n in S and S[n]
                   and (r.get("headroom") if "headroom" in r
                        else r["metric_orig"] - r["metric_base"]) >= args.min_headroom
                   and all(f"e{round(t*100):02d}" in r["variants"] for t in TAUS))
    print(f"{args.label}: {len(names)} adapters")

    def U(n, lab):
        v = task[n]["variants"][lab]
        if v.get("retained") is not None:
            return float(v["retained"])
        fl = task[n]["metric_base"]
        return (v["metric"] - fl) / (task[n]["metric_orig"] - fl)

    levels = {n: sorted([dict(lab=l, k=int(round(task[n]["variants"][l]["rank_frac"]
                                                 * args.nominal)),
                              L=max(task[n]["variants"][l]["L_W"], 1e-9), u=U(n, l))
                         for l in task[n]["variants"]], key=lambda r: -r["k"])
              for n in names}

    def allocate(gamma, budget):
        def at(level):
            out = {}
            for n in names:
                best = levels[n][0]
                for row in levels[n]:
                    if math.log(row["L"]) + gamma * math.log(S[n]) <= level:
                        best = row
                out[n] = best
            return out
        lo, hi = -60.0, 40.0
        for _ in range(90):
            m = 0.5 * (lo + hi)
            if sum(v["k"] for v in at(m).values()) > budget:
                lo = m
            else:
                hi = m
        a = at(hi)
        while sum(v["k"] for v in a.values()) > budget:
            n = max(names, key=lambda m: a[m]["k"])
            nxt = [r for r in levels[n] if r["k"] < a[n]["k"]]
            if not nxt:
                break
            a[n] = max(nxt, key=lambda r: r["k"])
        return a

    rules = [(f"gamma={g:g}" + ("  (uniform tau)" if g == 0 else
                                "  (P = S L_W, no fit)" if g == 1 else ""), g)
             for g in args.gammas]
    for spec_s in args.named:
        nm, val = spec_s.rsplit("=", 1)
        rules.append((nm, float(val)))

    out = {}
    for t0 in (0.90, 0.80, 0.70, 0.50):
        lab0 = f"e{round(t0*100):02d}"
        budget = sum(next(r for r in levels[n] if r["lab"] == lab0)["k"] for n in names)
        print(f"\n=== budget {budget} (uniform tau={t0}) ===")
        print(f"{'rule':30s}{'mean':>8}{'worst':>8}{'p10':>8}{'broken':>8}{'u<0.5':>7}")
        out[budget] = {}
        for nm, g in rules:
            a = allocate(g, budget)
            u = np.array([a[n]["u"] for n in names])
            out[budget][nm] = dict(gamma=g, mean=float(u.mean()), worst=float(u.min()),
                                   p10=float(np.percentile(u, 10)),
                                   broken=int((u <= 0).sum()),
                                   below05=int((u < 0.5).sum()))
            print(f"{nm:30s}{u.mean():8.3f}{u.min():8.3f}{np.percentile(u,10):8.3f}"
                  f"{int((u<=0).sum()):8d}{int((u<0.5).sum()):7d}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"pool": args.label, "n": len(names),
                                       "budgets": out}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
