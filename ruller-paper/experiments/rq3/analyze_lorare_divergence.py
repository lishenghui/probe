#!/usr/bin/env python3
"""Fit Table 5 regressions for completed LoRARetriever divergence shards."""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np


def fit(rows: list[dict], target: str, columns: tuple[str, ...]):
    y = np.log([r[target] for r in rows])
    x = np.column_stack([[np.log(r[c]) for r in rows] for c in columns]
                        + [np.ones(len(rows))])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    r2 = 1 - np.square(y - x @ beta).sum() / np.square(y - y.mean()).sum()
    return beta, r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/rq3/results/lorare_div_s*.json")
    args = ap.parse_args()
    rows = []
    for filename in sorted(glob.glob(args.results)):
        for record in json.load(open(filename)):
            for tau, value in record["variants"].items():
                rows.append(dict(adapter=record["short"], tau=tau, S=record["S"],
                                 L_W=value["L_W"], d_prompt=value["d_prompt"],
                                 d_out=value["d_out"]))
    adapters = sorted({r["adapter"] for r in rows})
    print(f"{len(rows)} cells over {len(adapters)} adapters")
    for target in ("d_prompt", "d_out"):
        both, r2 = fit(rows, target, ("S", "L_W"))
        _, r2s = fit(rows, target, ("S",))
        _, r2l = fit(rows, target, ("L_W",))
        loo = [fit([r for r in rows if r["adapter"] != adapter], target,
                   ("S", "L_W"))[1] for adapter in adapters]
        print(f"{target:8s}: R2(S)={r2s:.3f} R2(L_W)={r2l:.3f} "
              f"R2(both)={r2:.3f} a={both[0]:.2f} b={both[1]:.2f} "
              f"LOAO=[{min(loo):.3f},{max(loo):.3f}]")


if __name__ == "__main__":
    main()
