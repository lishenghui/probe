#!/usr/bin/env python3
"""Main figure: equal spectral truncation is not equal functional perturbation.

Left  -- functional divergence against adapter strength, log-log, one fit per
         truncation level.  The slope steepens as truncation gets aggressive.
Right -- divergence rescaled by the squared effective perturbation
          P = S*sqrt(1-tau).  The vertical spread collapses.

The retained-rank control used in the long paper is deliberately omitted here:
the short-paper mechanism figure needs readable evidence for the failure and
the calibrated collapse, while the diagnostic belongs in the appendix.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEVELS = [("e99", 0.99, "#B8B8B8", "o"), ("e95", 0.95, "#3F79B7", "s"),
          ("e90", 0.90, "#B00072", "^")]


def fit(x, y):
    b, a = np.polyfit(np.log(x), np.log(y), 1)
    r = np.corrcoef(np.log(x), np.log(y))[0, 1]
    return b, a, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--lw", type=Path, required=True,
                    help="cts_scaling_intervention.json, for the achieved L_W")
    ap.add_argument("--strengths", type=Path, required=True,
                    help="cts_strength_aggregations.json, for the global-norm S")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    LW = {r["adapter"]: r for r in json.loads(args.lw.read_text())["L_W"]}
    agg = {r["adapter"]: r for r in json.loads(args.strengths.read_text())}
    rows = [r for r in json.loads(args.sweep.read_text())
            if r["adapter"] in LW and r["adapter"] in agg]
    # The global concatenated norm, so that P = S * L_W is the exact
    # model-relative residual rather than a mean of per-module ratios.
    S = np.array([agg[r["adapter"]]["S_global"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    plt.rcParams.update({"font.size": 8})

    ax = axes[0]
    for label, tau, colour, marker in LEVELS:
        d = np.array([r["variants"][label]["d_js_mean"] for r in rows])
        b, a, r = fit(S, d)
        ax.scatter(S, d, s=15, c=colour, marker=marker, alpha=.75, linewidths=0,
                   label=fr"$\tau={tau:.2f}$: $S^{{{b:.2f}}}$, $\rho_{{\log}}={r:.2f}$")
        xs = np.array([S.min(), S.max()])
        ax.plot(xs, np.exp(a) * xs ** b, c=colour, lw=1.1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("adapter strength $S$"); ax.set_ylabel(r"$D_{\mathrm{JS}}$ (nats)")
    ax.set_title("(a) divergence vs. strength", fontsize=9)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")

    # (b) The two exponents are not equal, so P = S*L_W is not the coordinate the
    # points collapse onto; S^(a/b) L_W is.  Plot against the calibrated
    # coordinate and report both fits so the difference is visible.
    ax = axes[1]
    # joint fit gives the exponent ratio that defines the calibrated coordinate
    lS, lL, lD = [], [], []
    for label, tau, colour, marker in LEVELS:
        for i, r_ in enumerate(rows):
            lS.append(np.log(S[i])); lL.append(np.log(LW[r_["adapter"]][f"L_{label}"]))
            lD.append(np.log(r_["variants"][label]["d_js_mean"]))
    A = np.column_stack([np.ones(len(lD)), lS, lL])
    coef, *_ = np.linalg.lstsq(A, np.array(lD), rcond=None)
    a_hat, b_hat = coef[1], coef[2]
    allX, allD = [], []
    for label, tau, colour, marker in LEVELS:
        X = np.array([S[i] ** (a_hat / b_hat) * LW[rows[i]["adapter"]][f"L_{label}"]
                      for i in range(len(rows))])
        d = np.array([r_["variants"][label]["d_js_mean"] for r_ in rows])
        ax.scatter(X, d, s=15, c=colour, marker=marker, alpha=.75, linewidths=0,
                   label=fr"$\tau={tau:.2f}$")
        allX.append(X); allD.append(d)
    X = np.concatenate(allX); d = np.concatenate(allD)
    bb, aa, r = fit(X, d)
    xs = np.array([X.min(), X.max()])
    ax.plot(xs, np.exp(aa) * xs ** bb, c="#333", lw=1.2, zorder=1)
    P = np.concatenate([np.array([S[i] * LW[rows[i]["adapter"]][f"L_{lab}"]
                                  for i in range(len(rows))]) for lab, _, _, _ in LEVELS])
    _, _, rP = fit(P, d)
    ax.text(.04, .95,
            fr"$\hat a={a_hat:.2f}$, $\hat b={b_hat:.2f}$, $\hat a/\hat b={a_hat/b_hat:.2f}$"
            "\n" fr"calibrated: $R^2={r*r:.2f}$" "\n" fr"$P=S L_W$: $R^2={rP*rP:.2f}$",
            transform=ax.transAxes, fontsize=6.8, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"calibrated residual $S^{\hat a/\hat b}L_W$")
    ax.set_ylabel(r"$D_{\mathrm{JS}}$ (nats)")
    ax.set_title("(b) calibrated two-factor collapse", fontsize=9)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")

    for label, tau, _, _ in LEVELS:
        d = np.array([r["variants"][label]["d_js_mean"] for r in rows])
        f = np.array([r["variants"][label]["rank_frac"] for r in rows])
        b, _, r = fit(S, d)
        n = len(rows); order = np.argsort(S)
        lo, hi = order[: n // 3], order[2 * n // 3:]
        print(f"{label}: D~S^{b:.2f} r={r:.3f} | tercile {np.median(d[hi])/np.median(d[lo]):.1f}x "
              f"| rank frac {np.median(f[lo]):.2f}->{np.median(f[hi]):.2f} "
              f"(slope S^{fit(S, f)[0]:+.2f})")


if __name__ == "__main__":
    main()
