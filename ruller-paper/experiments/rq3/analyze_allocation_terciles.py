#!/usr/bin/env python3
"""How much utility strength-calibrated allocation moves, and from whom.

Whether SCT helps cannot be settled by asking whether the average adapter
improves: it is a minimax rule, so it is built to make most adapters slightly
worse. Twelve paired tests over three pools and four budgets duly return nothing.

Nor can it be settled on the outcome that motivated the method -- an adapter
driven below the un-adapted model -- because that is a rare event. Across every
pool, budget and adapter measured here, only three adapters are ever in that
state, and no reweighting of a three-event outcome is estimable.

What is estimable is the transfer itself, on a continuous outcome and the full
sample. Terciles are taken *within* each pool so the contrast is strength rather
than provenance, and the bootstrap is clustered by adapter because each adapter
contributes one row per budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True,
                    help="per (adapter, budget) rows with S, du, produced by the "
                         "grid allocation over each pool")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    R = json.loads(args.rows.read_text())
    keys = sorted({(r["pool"], r["adapter"]) for r in R})
    S = {k: next(r["S"] for r in R if (r["pool"], r["adapter"]) == k) for k in keys}

    grp = {}
    for pool in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == pool]
        lo, hi = np.percentile([S[k] for k in ks], [33.3, 66.7])
        for k in ks:
            grp[k] = "low" if S[k] <= lo else ("mid" if S[k] <= hi else "high")

    by = {k: [r["du"] for r in R if (r["pool"], r["adapter"]) == k] for k in keys}

    def boot(ks):
        obs = float(np.mean([x for k in ks for x in by[k]]))
        d = np.array([np.mean([x for k in [ks[i] for i in rng.integers(0, len(ks), len(ks))]
                               for x in by[k]]) for _ in range(args.draws)])
        return obs, np.percentile(d, [2.5, 97.5]), float((d > 0).mean())

    print(f"{len(keys)} adapters, {len(R)} (adapter, budget) rows\n")
    print(f"{'tercile':>8}{'adapters':>10}{'mean du':>10}{'95% CI':>22}{'P(>0)':>8}")
    for g in ("low", "mid", "high"):
        ks = [k for k in keys if grp[k] == g]
        obs, ci, p = boot(ks)
        print(f"{g:>8}{len(ks):10d}{obs:+10.4f}   [{ci[0]:+.4f},{ci[1]:+.4f}]{p*100:7.1f}%")

    kh = [k for k in keys if grp[k] == "high"]
    kl = [k for k in keys if grp[k] == "low"]
    d = np.array([np.mean([x for k in [kh[i] for i in rng.integers(0, len(kh), len(kh))]
                           for x in by[k]])
                  - np.mean([x for k in [kl[i] for i in rng.integers(0, len(kl), len(kl))]
                             for x in by[k]]) for _ in range(args.draws)])
    obs = (np.mean([x for k in kh for x in by[k]])
           - np.mean([x for k in kl for x in by[k]]))
    ci = np.percentile(d, [2.5, 97.5])
    print(f"\nhigh - low: {obs:+.4f}   95% CI [{ci[0]:+.4f},{ci[1]:+.4f}]   "
          f"P(>0) = {(d > 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
