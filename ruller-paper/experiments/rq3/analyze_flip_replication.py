#!/usr/bin/env python3
"""Replicate the flip-prediction mechanism on a second adapter population.

Sec. 5.7 was the paper's last single-population result: perturbation magnitude and
the uncompressed model's local decision margin jointly predict a token flip on
seven LoRA Land adapters. Everything around it now spans several pools, which made
one pool conspicuous rather than sufficient.

The controlled pool's per-token dumps already exist, from the output-position run
re-measured after the prompt truncation fix, so the identical specification runs
on 30 adapters with no new generation.

Two choices carry over from analyze_margin_probe.py and matter:

  * the perturbation regressor is the *cell-level* D_out, one number per
    (adapter, tau). Using the per-token J_t is circular -- a flip at t forces a
    large J_t at t -- and scores ~.98 while meaning nothing. That row is printed
    only as the circularity check it is.
  * the fold is the adapter, so the held-out adapter contributes neither to the
    logistic fit nor, in the closed-loop row, to the law that predicts its own
    divergence.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", nargs="+", required=True, help="*_tokens directories")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--strengths", type=Path, required=True)
    ap.add_argument("--label", default="pool")
    args = ap.parse_args()

    tok = []
    for pat in args.tokens:
        for d in sorted(glob.glob(pat)):
            for f in sorted(glob.glob(d + "/*.npz")):
                m = re.match(r"(.+)-e(\d+)\.npz", Path(f).name)
                if not m:
                    continue
                z = np.load(f)
                tok.append(dict(adapter=m.group(1), tau=int(m.group(2)) / 100,
                                M=z["per_token_orig_margin"],
                                Mc=z["per_token_comp_margin"],
                                J=z["per_token_js"]))
    S = {e["adapter"]: e.get("S_global", e.get("S"))
         for e in json.loads(args.strengths.read_text())}
    rows = []
    for pat in args.results:
        for f in sorted(glob.glob(pat)):
            for r in json.loads(Path(f).read_text()):
                for lab, v in r["variants"].items():
                    if v.get("d_out") and v.get("L_W") and S.get(r["adapter"]):
                        rows.append(dict(adapter=r["adapter"], tau=int(lab[1:]) / 100,
                                         d_out=v["d_out"], L_W=v["L_W"],
                                         S=S[r["adapter"]]))
    d_out = {(r["adapter"], r["tau"]): r["d_out"] for r in rows}

    X, y, grp, cell = [], [], [], []
    for t in tok:
        k = (t["adapter"], t["tau"])
        if k not in d_out:
            continue
        ok = (t["M"] > 0) & (t["J"] > 0)          # margin and divergence both defined
        if not ok.sum():
            continue
        X.append(np.column_stack([np.full(ok.sum(), np.log(d_out[k])),
                                  np.log(t["M"][ok]), np.log(t["J"][ok])]))
        y.append((t["Mc"][ok] < 0).astype(int))
        grp.append(np.full(ok.sum(), t["adapter"]))
        cell += [k] * int(ok.sum())
    X, y, grp = np.vstack(X), np.concatenate(y), np.concatenate(grp)
    folds = sorted(set(grp))
    print(f"{args.label}: {len(folds)} adapters, {len(y)} tokens, "
          f"flip rate {y.mean():.3%}")

    def auroc(cols):
        sc = {}
        for a in folds:
            tr, te = grp != a, grp == a
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            lr = LogisticRegression(max_iter=2000).fit(X[tr][:, cols], y[tr])
            sc[a] = roc_auc_score(y[te], lr.decision_function(X[te][:, cols]))
        return float(np.mean(list(sc.values()))), len(sc)

    print(f"\n{'specification':34s}{'AUROC':>8}{'folds':>7}")
    for nm, c in (("D_out only", [0]), ("margin only", [1]),
                  ("D_out + margin", [0, 1]),
                  ("J_t at the same token (circular)", [2])):
        a, n = auroc(c)
        print(f"{nm:34s}{a:8.3f}{n:7d}")

    sc = {}
    for a in folds:
        tr, te = grp != a, grp == a
        train = [r for r in rows if r["adapter"] != a]
        if (len(set(y[tr])) < 2 or len(set(y[te])) < 2
                or len({r["adapter"] for r in train}) < 2):
            continue
        A = np.array([[math.log(r["S"]), math.log(r["L_W"]), 1.0] for r in train])
        be = np.linalg.lstsq(A, np.array([math.log(r["d_out"]) for r in train]),
                             rcond=None)[0]
        hat = {(r["adapter"], r["tau"]):
               be[0] * math.log(r["S"]) + be[1] * math.log(r["L_W"]) + be[2]
               for r in rows}
        Xh = np.column_stack([np.array([hat[k] for k in cell]), X[:, 1]])
        lr = LogisticRegression(max_iter=2000).fit(Xh[tr], y[tr])
        sc[a] = roc_auc_score(y[te], lr.decision_function(Xh[te]))
    print(f"{'pred D_out(S,L_W) + margin':34s}"
          f"{float(np.mean(list(sc.values()))):8.3f}{len(sc):7d}")


if __name__ == "__main__":
    main()
