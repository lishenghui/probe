#!/usr/bin/env python3
"""Does the primary-pool conclusion survive un-truncated prompts?

Merges the shards of cts_prompt_sensitivity.py and re-derives, under both
tokenisations, the three numbers the paper actually leans on:

  * the two-factor fit          log D = c + a log S + b log L_W
  * what L_W explains alone     (the complementarity claim)
  * the strength-tercile ratio  (the headline 52.8x at tau = 0.90)

The old setting truncates 21 of 30 adapters' prompts down to a shared preamble,
so its 48 prompts per adapter are one prompt repeated; `unique` below is that
count and is the direct evidence.  What matters is not that the old numbers were
mis-computed -- they were computed correctly on a degenerate input -- but whether
the conclusion moves once the input contains the task again.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

SETTINGS = ("old", "fixed")


def fit(pts):
    X = np.column_stack([np.log([p[0] for p in pts]), np.log([p[1] for p in pts]),
                         np.ones(len(pts))])
    y = np.log([p[2] for p in pts])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    b1, *_ = np.linalg.lstsq(X[:, 1:], y, rcond=None)
    r2l = 1 - ((y - X[:, 1:] @ b1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    Xs = np.column_stack([np.log([p[0] for p in pts]), np.ones(len(pts))])
    b2, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    r2s = 1 - ((y - Xs @ b2) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, r2, r2l, r2s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--lw", type=Path,
                    default=Path("artifacts/rq3/results/cts_scaling_intervention.json"))
    ap.add_argument("--tau", default="e90", help="threshold for the tercile table")
    args = ap.parse_args()

    recs = []
    for pat in args.results:
        for f in sorted(glob.glob(pat)):
            recs.extend(json.loads(Path(f).read_text()))
    if not recs:
        print("no shards found")
        return
    recs.sort(key=lambda r: r["S"])
    lw = {r["adapter"]: r for r in json.loads(args.lw.read_text())["L_W"]} \
        if args.lw.is_file() else {}
    print(f"{len(recs)} adapters, S from {recs[0]['S']:.4f} to {recs[-1]['S']:.4f}")

    taus = sorted(recs[0]["settings"]["old"]["variants"], reverse=True)
    print(f"\n{'adapter':10s} {'S':>7} | " +
          " | ".join(f"{s:>5} uniq {'  '.join(t for t in taus)}" for s in SETTINGS))
    for r in recs:
        line = f"{r['adapter']:10s} {r['S']:7.4f} |"
        for s in SETTINGS:
            p = r["settings"][s]
            line += f" {p['unique_prompts']:4d}/{r['prompts']:<3d}" + \
                    "".join(f" {p['variants'][t]['d_js_mean']:9.2e}" for t in taus) + " |"
        print(line)

    print(f"\n--- two-factor fit ---")
    print(f"{'setting':8s} {'n':>4} {'R2(S)':>7} {'R2(L_W)':>8} {'R2(S,L_W)':>10} "
          f"{'a':>6} {'b':>6} {'a-b':>7}")
    for s in SETTINGS:
        pts = [(r["S"], lw[r["adapter"]][f"L_{t}"], v["d_js_mean"])
               for r in recs if r["adapter"] in lw
               for t, v in r["settings"][s]["variants"].items()]
        if len(pts) < 6:
            print(f"{s:8s} too few points"); continue
        beta, r2, r2l, r2s = fit(pts)
        print(f"{s:8s} {len(pts):4d} {r2s:7.3f} {r2l:8.3f} {r2:10.3f} "
              f"{beta[0]:6.2f} {beta[1]:6.2f} {beta[0] - beta[1]:7.2f}")

    print(f"\n--- strength terciles at {args.tau} (the headline ratio) ---")
    n = len(recs)
    cut = [recs[:n // 3], recs[n // 3:2 * n // 3], recs[2 * n // 3:]]
    print(f"{'tercile':8s} {'n':>3} {'median S':>9} " +
          " ".join(f"{s + ' D_JS':>12} {s + ' flip':>10}" for s in SETTINGS))
    hi_lo = {}
    for lab, grp in zip(("low", "mid", "high"), cut):
        row = f"{lab:8s} {len(grp):3d} {np.median([r['S'] for r in grp]):9.4f} "
        for s in SETTINGS:
            d = np.median([r["settings"][s]["variants"][args.tau]["d_js_mean"] for r in grp])
            fl = np.median([r["settings"][s]["variants"][args.tau]["flip"] for r in grp])
            hi_lo.setdefault(s, []).append(d)
            row += f" {d:12.3e} {fl:10.2%}"
        print(row)
    print(f"{'high/low':8s} {'':3s} {'':9s} " +
          " ".join(f" {hi_lo[s][2] / max(hi_lo[s][0], 1e-30):11.1f}x {'':10s}"
                   for s in SETTINGS))


if __name__ == "__main__":
    main()
