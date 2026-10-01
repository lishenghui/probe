#!/usr/bin/env python3
"""Does the margin mechanism generalise from LoRA Land to the primary pool?

Sec. 4.6 established, on seven adapters with recoverable task metrics, that
output perturbation alone does not decide whether a decision flips: the
uncompressed model's margin at that decision does the rest, and a divergence
predicted from (S, L_W) plus that margin reaches leave-one-adapter-out AUROC
.953.  Seven adapters from one release is a thin basis for a mechanism claim.

This repeats it on the 32-adapter Lots-of-LoRAs pool -- different base model,
rank, target modules and training pipeline -- using the same shared-prefix probe
and the same two specifications, the honest one on cell-level D_out and the
circular one on J_t, reported together so the inflation stays visible.

A power caveat is built in.  At the three thresholds the paper reports, this pool
barely flips anything: 16 of 30 adapters produce no flipped token at all.  The
sweep therefore extends to tau = .80, .70, .50, and adapters that still produce
no positives are excluded from the AUROC with the exclusion reported rather than
silently dropped.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_margin_probe import _law, logistic_check  # noqa: E402

TAU_ORDER = {"e99": 6, "e95": 5, "e90": 4, "e80": 3, "e70": 2, "e50": 1}


def load(results: list[Path], lw_file: Path | None, strength_file: Path | None):
    """Rows and per-token probes, with L_W back-filled if the run predates it."""
    lw = {}
    if lw_file and lw_file.is_file():
        lw = {r["adapter"]: r for r in json.loads(lw_file.read_text())["L_W"]}
    global_strength = {}
    if strength_file and strength_file.is_file():
        global_strength = {
            r["adapter"]: r["S_global"] for r in json.loads(strength_file.read_text())
        }
    rows, tok = [], []
    for res in results:
        tdir = res.with_suffix("").with_name(res.stem + "_tokens")
        for r in json.loads(res.read_text()):
            for tau, v in r["variants"].items():
                L = v.get("L_W") or lw.get(r["adapter"], {}).get(f"L_{tau}")
                if L is None:
                    continue
                rows.append(dict(adapter=r["adapter"], tau=tau,
                                 S=global_strength.get(r["adapter"], r["S"]), L_W=L,
                                 d_prompt=v["d_prompt"], d_out=v["d_out"],
                                 flip=v["flip_out"], split=r.get("split", "?"),
                                 npos=v["out_positions"]))
                f = tdir / f"{r['adapter']}-{tau}.npz"
                if f.is_file():
                    z = np.load(f)
                    tok.append(dict(adapter=r["adapter"], tau=tau,
                                    M=z["per_token_orig_margin"].astype(np.float64),
                                    Mc=z["per_token_comp_margin"].astype(np.float64),
                                    J=z["per_token_js"].astype(np.float64),
                                    ex=z["example"], step=z["step"]))
    return rows, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--lw", type=Path,
                    default=Path("artifacts/rq3/results/cts_scaling_intervention.json"))
    ap.add_argument("--strengths", type=Path,
                    default=Path("artifacts/rq3/results/cts_strength_aggregations.json"),
                    help="Global Frobenius S values used by the paper; falls back "
                         "to the legacy modulewise mean embedded in each shard")
    args = ap.parse_args()

    files = [Path(f) for pat in args.results for f in sorted(glob.glob(str(pat)))]
    rows, tok = load(files, args.lw, args.strengths)
    if not rows:
        print("no rows loaded")
        return
    ads = sorted({r["adapter"] for r in rows})
    print(f"{len(rows)} cells over {len(ads)} adapters, split="
          f"{ {r['split'] for r in rows} }, {len(tok)} token probes")

    print("\n--- strength law on this pool ---")
    for key in ("d_prompt", "d_out"):
        sub = [r for r in rows if r.get(key)]
        if len(sub) < 6:
            continue
        beta, r2 = _law(sub, key)
        X = np.column_stack([np.log([r["L_W"] for r in sub]), np.ones(len(sub))])
        y = np.log([r[key] for r in sub])
        b1, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2l = 1 - ((y - X @ b1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {key:9s} n={len(sub):3d}  a={beta[0]:+.2f} b={beta[1]:+.2f}  R2={r2:.3f}"
              f"   (L_W alone {r2l:.3f}, S adds {r2 - r2l:+.3f})")

    print(f"\n--- decision geometry ---")
    print(f"{'adapter':10s} {'S':>7} {'T':>6} {'M p10':>6} {'M p50':>6} " +
          " ".join(f"{t:>7}" for t in sorted(TAU_ORDER, key=TAU_ORDER.get, reverse=True)))
    flips_by_adapter = {}
    for a in ads:
        ts = [t for t in tok if t["adapter"] == a]
        if not ts:
            continue
        M = np.concatenate([t["M"] for t in ts])
        T = ts[0]["ex"].size / max(np.unique(ts[0]["ex"]).size, 1)
        cells = {t["tau"]: float((t["Mc"] < 0).mean()) for t in ts}
        flips_by_adapter[a] = sum(int((t["Mc"] < 0).sum()) for t in ts)
        print(f"{a:10s} {ts[0].get('S', 0) or 0:7.4f} {T:6.1f} "
              f"{np.percentile(M, 10):6.2f} {np.median(M):6.2f} " +
              " ".join(f"{cells.get(t, float('nan')):7.2%}"
                       for t in sorted(TAU_ORDER, key=TAU_ORDER.get, reverse=True)))
    v = np.array(sorted(flips_by_adapter.values()))
    print(f"\nflipped tokens per adapter: median {np.median(v):.0f}  max {v.max()}  "
          f"with >=10: {(v >= 10).sum()}/{len(v)}  with 0: {(v == 0).sum()}/{len(v)}")

    res, X, y, grp = logistic_check(tok, rows)
    names = sorted(set(grp))
    ev = [a for a in names if any(a in sc for _, sc in res)]
    print(f"\nleave-one-adapter-out AUROC for P(flip)  (n={len(y):,} tokens, "
          f"{y.mean():.2%} positive; {len(ev)}/{len(names)} adapters evaluable)")
    print(f"  {'model':16s} {'mean':>7} {'median':>7} {'folds':>6}")
    for name, sc in res:
        vals = list(sc.values())
        if not vals:
            print(f"  {name:16s} {'-':>7} {'-':>7} {0:6d}")
            continue
        print(f"  {name:16s} {np.mean(vals):7.3f} {np.median(vals):7.3f} {len(vals):6d}")


if __name__ == "__main__":
    main()
