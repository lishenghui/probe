#!/usr/bin/env python3
"""Two-layer analysis of the Lora Land truncation sweep.

Layer 1, already established elsewhere, is (S, L_W) -> D_JS.  This script is
about layer 2: D_JS -> task damage, which has no universal mapping.  ViGGO
churns 73/200 outputs at D_JS = 0.001 while WikiSQL absorbs D_JS = 0.27 without
losing a point, a ~270x spread in how much behavioural movement a task can take.

The mechanism is decomposed per generated token, under a shared original prefix
(see margin_probe in predibase_task_metrics.py):

    E_t = M_t - ~M_t   how far compression pushed this decision
    M_t                how far it had to push before the argmax changes
    flip <=> E_t > M_t

so E is the perturbation side and M is the task side.  Note that the normalised
form R_t = E_t / M_t obeys "R_t > 1 iff flip" *by definition*, not as a finding;
the content is whether E is predicted by the perturbation alone while M varies
by task, which is what the figures test.

The smoke run turned up a second task-side term that margin alone does not
cover: exposure.  SST-2 decides 4 tokens per example and GSM8K decides ~110, so
at e50 GSM8K flips only 7.9% of tokens yet 100% of sequences.  Susceptibility is
therefore per-decision robustness (M) compounded over the number of decisions
(T), and the table below reports the independent-flip baseline 1-(1-p)^T so that
the two can be told apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SERIES = ("#2a78d6", "#eb6834")            # validated all-pairs, light surface
SEQ = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
BLUES = LinearSegmentedColormap.from_list("blues", SEQ)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def load(results: list[Path]):
    """Merge several result files; later files only fill gaps left by earlier ones.

    d_prompt is what divergence() in predibase_task_metrics.py reports: a JS
    averaged over *prompt* positions, i.e. over the conditioning, never over a
    token the deployed model actually emits.  d_out is the same divergence
    restricted to the generated decisions, taken from the shared-prefix probe.
    They are not interchangeable and the table prints both.
    """
    rows, tok, seen = [], [], {}
    # a probe can land before the json that names it (results are rewritten per
    # adapter, npz per variant), so look in every companion directory
    tok_dirs = [res.with_suffix("").with_name(res.stem + "_tokens") for res in results]
    for res in results:
        for r in json.loads(res.read_text()):
            n = r["n"]
            for tau, v in r["variants"].items():
                key = (r["adapter"], tau)
                f = next((d / f"{r['adapter']}-{tau}.npz" for d in tok_dirs
                          if (d / f"{r['adapter']}-{tau}.npz").is_file()), tok_dirs[0])
                if key in seen and not (f.is_file() and seen[key]["d_out"] is None):
                    continue
                row = dict(adapter=r["adapter"], tau=tau, S=r["S"], L_W=v["L_W"],
                           P=v["P"], d_prompt=v["d_js"], d_out=None, n=n,
                           metric_orig=r["metric_orig"], metric=v["metric"],
                           c2w=v.get("c2w", 0), w2c=v.get("w2c", 0),
                           churn=(v.get("c2w", 0) + v.get("w2c", 0)) / n,
                           net=(v["metric"] - r["metric_orig"]) / max(r["metric_orig"], 1e-9))
                if f.is_file():
                    z = np.load(f)
                    t = dict(adapter=r["adapter"], tau=tau, d_prompt=v["d_js"],
                             M=z["per_token_orig_margin"].astype(np.float64),
                             Mc=z["per_token_comp_margin"].astype(np.float64),
                             J=z["per_token_js"].astype(np.float64),
                             ex=z["example"], step=z["step"])
                    row["d_out"] = float(t["J"].mean())
                    tok = [x for x in tok if (x["adapter"], x["tau"]) != key] + [t]
                if key in seen:
                    rows[seen[key]["i"]] = row
                else:
                    seen[key] = dict(i=len(rows), d_out=row["d_out"])
                    rows.append(row)
                seen[key]["d_out"] = row["d_out"]
    return rows, tok


def _label_ends(ax, items):
    """Direct labels at each curve's end, pushed apart so seven series stay legible."""
    items = sorted(items, key=lambda it: it[1])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.075
    for i in range(1, len(items)):
        x, y, t = items[i]
        if y - items[i - 1][1] < gap:
            items[i] = (x, items[i - 1][1] + gap, t)
    for (x, y, t), (x0, y0, _) in zip(items, sorted(items, key=lambda it: it[1])):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(5, -2),
                    fontsize=6.5, color=INK2, annotation_clip=False)


def fig_perturbation_vs_utility(rows, out: Path):
    """Left: the divergence the paper used vs the divergence at the decisions it
    is meant to describe.  Right: layer 2 -- at matched output perturbation the
    task consequence still spans two orders of magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), constrained_layout=True)
    adapters = sorted({r["adapter"] for r in rows})
    probed = [r for r in rows if r.get("d_out")]

    ax = axes[0]
    lim = [min(min(r["d_out"] for r in probed), min(r["d_prompt"] for r in probed)) * 0.6,
           max(max(r["d_out"] for r in probed), max(r["d_prompt"] for r in probed)) * 1.6]
    ax.plot(lim, lim, ls="--", lw=0.8, color=MUTED, zorder=1)
    ends = []
    for a in adapters:
        sub = sorted((r for r in probed if r["adapter"] == a), key=lambda r: r["d_out"])
        x = [r["d_out"] for r in sub]
        y = [r["d_prompt"] for r in sub]
        ax.plot(x, y, "-", color=SERIES[0], lw=1.0, alpha=0.55, zorder=2)
        ax.plot(x, y, "o", color=SERIES[0], ms=3.5, mec="#fcfcfb", mew=0.6, zorder=3)
        ends.append((x[-1], y[-1], a))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"$D_{\mathrm{out}}$  (generated decisions)")
    ax.set_ylabel(r"$D_{\mathrm{prompt}}$  (conditioning prefixes)")
    ax.annotate("prompt divergence\noverstates by up to $52\\times$", (lim[0] * 3, lim[1] * 0.3),
                fontsize=6.5, color=INK2)
    for x, y, t in ends:
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, 2),
                    fontsize=6.5, color=INK2)

    ax = axes[1]
    ends = []
    for a in adapters:
        sub = sorted((r for r in probed if r["adapter"] == a), key=lambda r: r["d_out"])
        x = [r["d_out"] for r in sub]
        y = [r["churn"] for r in sub]
        ax.plot(x, y, "-", color=SERIES[0], lw=1.0, alpha=0.55, zorder=2)
        ax.plot(x, y, "o", color=SERIES[0], ms=3.5, mec="#fcfcfb", mew=0.6, zorder=3)
        ends.append((x[-1], y[-1], a))
    ax.set_xscale("log")
    ax.set_xlabel(r"$D_{\mathrm{out}}$  (generated decisions)")
    ax.set_ylabel("outputs changed (fraction of $n$)")
    _label_ends(ax, ends)

    for ax in axes:
        ax.grid(True, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


TAU_ORDER = {"e99": 6, "e95": 5, "e90": 4, "e80": 3, "e70": 2, "e50": 1}


def _law(sub, key="d_out"):
    """Least squares for log D = c + a log S + b log L_W; returns (beta, R2)."""
    X = np.column_stack([np.log([r["S"] for r in sub]), np.log([r["L_W"] for r in sub]),
                         np.ones(len(sub))])
    y = np.log([r[key] for r in sub])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, r2


def fit_law(rows):
    """Refit the strength law on both divergences.

    The law was established against the prompt-position divergence.  If it only
    holds there, it predicts how the conditioning distribution moves rather than
    how the deployed model's own decisions move, and the paper has to say so.
    """
    print("\n--- strength law: log D = c + a log S + b log L_W ---")
    probed = [r for r in rows if r.get("d_out")]
    # the middle row refits d_prompt on exactly the probed cells, so the D_out
    # comparison is not just a different (adapter, tau) sample
    for label, key, sub in (("d_prompt", "d_prompt", [r for r in rows if r.get("d_prompt")]),
                            ("d_prompt|", "d_prompt", probed),
                            ("d_out", "d_out", probed)):
        if len(sub) < 6:
            print(f"  {label:9s} n={len(sub)}: too few points")
            continue
        beta, r2 = _law(sub, key)
        X = np.column_stack([np.log([r["S"] for r in sub]),
                             np.log([r["L_W"] for r in sub]),
                             np.ones(len(sub))])
        y = np.log([r[key] for r in sub])
        # Both single-variable fits, matching every column of Table tab:where.
        bs, *_ = np.linalg.lstsq(X[:, [0, 2]], y, rcond=None)
        r2_s = 1 - ((y - X[:, [0, 2]] @ bs) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        b1, *_ = np.linalg.lstsq(X[:, 1:], y, rcond=None)
        r2_l = 1 - ((y - X[:, 1:] @ b1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {label:9s} n={len(sub):3d}  a(S)={beta[0]:+.2f} b(L_W)={beta[1]:+.2f}  "
              f"R2={r2:.3f}   (S alone {r2_s:.3f}, L_W alone {r2_l:.3f}, "
              f"S adds {r2 - r2_l:+.3f})"
              f"   [{len({r['adapter'] for r in sub})} adapters]")


def buckets(v, k, log=False):
    q = np.quantile(np.log(v) if log else v, np.linspace(0, 1, k + 1))
    q[0], q[-1] = -np.inf, np.inf
    return np.clip(np.digitize(np.log(v) if log else v, q[1:-1]), 0, k - 1), q


def fig_flip_grid(tok, out: Path, k=4):
    """P(flip) over J x M.

    Kept as a diagnostic, deliberately not used in the paper: pooling across
    adapters whose margin distributions differ by an order of magnitude makes the
    M quartiles incomparable, and the top-J row comes out non-monotone in M
    (13.2%, 4.8%, 6.6%, 22.1%).  The leave-one-adapter-out table carries the same
    claim without that confound.  A per-adapter small-multiple version would be
    the honest figure.
    """
    M = np.concatenate([t["M"] for t in tok])
    Mc = np.concatenate([t["Mc"] for t in tok])
    J = np.concatenate([t["J"] for t in tok])
    ok = (M > 0) & (J > 0)
    M, Mc, J = M[ok], Mc[ok], J[ok]
    flip = Mc < 0
    bj, qj = buckets(J, k, log=True)
    bm, qm = buckets(M, k)
    grid = np.full((k, k), np.nan)
    cnt = np.zeros((k, k), int)
    for i in range(k):
        for j in range(k):
            sel = (bj == k - 1 - i) & (bm == j)
            cnt[i, j] = sel.sum()
            if sel.sum():
                grid[i, j] = flip[sel].mean()
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    im = ax.imshow(grid, cmap=BLUES, vmin=0, vmax=np.nanmax(grid), aspect="auto")
    for i in range(k):
        for j in range(k):
            if cnt[i, j]:
                ax.text(j, i, f"{grid[i, j]:.0%}\n$n$={cnt[i, j]:,}", ha="center",
                        va="center", fontsize=6,
                        color="#fcfcfb" if grid[i, j] > 0.55 * np.nanmax(grid) else INK)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Q{j+1}" for j in range(k)])
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Q{k-i}" for i in range(k)])
    ax.set_xlabel(r"original decision margin $M_t$  (quartile, low$\rightarrow$high)")
    ax.set_ylabel(r"token divergence $J_t$  (quartile)")
    ax.set_title("P(token flip)", loc="left", color=INK2)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).outline.set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, grid, cnt, np.exp(qj), qm


def fig_erosion_vs_margin(tok, out: Path, cap=40000, seed=0):
    """E is the perturbation side, M the task side, and flip is E > M -- so the
    per-adapter spread along M at matched E is the task-susceptibility term."""
    rng = np.random.default_rng(seed)
    adapters = sorted({t["adapter"] for t in tok})
    n = len(adapters)
    fig, axes = plt.subplots(1, n, figsize=(1.55 * n + 0.6, 2.5), constrained_layout=True,
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, adapters):
        M = np.concatenate([t["M"] for t in tok if t["adapter"] == a])
        Mc = np.concatenate([t["Mc"] for t in tok if t["adapter"] == a])
        keep = M > 0
        M, Mc = M[keep], Mc[keep]
        E = M - Mc
        if M.size > cap:
            idx = rng.choice(M.size, cap, replace=False)
            M, E = M[idx], E[idx]
        flip = E > M
        ax.scatter(M[~flip], E[~flip], s=1.5, c=SERIES[0], alpha=0.18, lw=0,
                   rasterized=True, label="held")
        ax.scatter(M[flip], E[flip], s=1.5, c=SERIES[1], alpha=0.45, lw=0,
                   rasterized=True, label="flipped")
        lim = np.nanpercentile(np.concatenate([M, E]), 99.5)
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--", zorder=4)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")          # so E = M reads as the 45-degree boundary
        ax.set_title(f"{a}\n{flip.mean():.1%} flip", loc="left", fontsize=7, color=INK2)
        ax.grid(True, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"erosion $E_t = M_t - \tilde{M}_t$")
    for ax in axes:
        ax.set_xlabel(r"$M_t$")
    h = [plt.Line2D([], [], marker="o", ls="", ms=4, color=c) for c in SERIES]
    fig.legend(h, ["held", r"flipped ($E_t > M_t$)"], loc="outside lower center",
               ncol=2, fontsize=7)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def logistic_check(tok, rows):
    """Does the local margin add anything over the perturbation?

    Three specifications, because the obvious one is circular.  A flip at t
    forces a large J_t at t -- the argmax moved, so the distributions must differ
    -- so "predict flip_t from J_t" scores ~0.99 by construction and means
    nothing.  The measured specification therefore uses the *cell-level* D_out,
    one number per (adapter, tau), which knows nothing about position t; all
    within-cell variation then has to come from M_t.  The closed-loop
    specification goes further and replaces even that with D_hat predicted from
    (S, L_W) alone, so nothing about the compressed model's behaviour is used.
    """
    d_out = {(r["adapter"], r["tau"]): r["d_out"] for r in rows if r.get("d_out")}
    X, y, grp, cell = [], [], [], []
    for t in tok:
        key = (t["adapter"], t["tau"])
        if key not in d_out:
            continue
        ok = (t["M"] > 0) & (t["J"] > 0)
        X.append(np.column_stack([np.full(ok.sum(), np.log(d_out[key])),
                                  np.log(t["M"][ok]), np.log(t["J"][ok])]))
        y.append((t["Mc"][ok] < 0).astype(int))
        grp.append(np.full(ok.sum(), t["adapter"]))
        cell.extend([key] * int(ok.sum()))
    X, y, grp = np.vstack(X), np.concatenate(y), np.concatenate(grp)
    folds = sorted(set(grp))
    out = []
    for name, cols in (("D_out only", [0]), ("M only", [1]), ("D_out + M", [0, 1]),
                       ("J_t (circular)", [2])):
        scores = {}
        for a in folds:
            tr, te = grp != a, grp == a
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            lr = LogisticRegression(max_iter=2000).fit(X[tr][:, cols], y[tr])
            scores[a] = roc_auc_score(y[te], lr.decision_function(X[te][:, cols]))
        out.append((name, scores))

    # Closed loop: the strength law is refit inside each fold, so the held-out
    # adapter never contributes to its own predicted divergence either.
    probed = [r for r in rows if r.get("d_out")]
    scores = {}
    for a in folds:
        tr, te = grp != a, grp == a
        train = [r for r in probed if r["adapter"] != a]
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2 or len({r["adapter"] for r in train}) < 2:
            continue
        beta, _ = _law(train)
        hat = {(r["adapter"], r["tau"]):
               beta[0] * np.log(r["S"]) + beta[1] * np.log(r["L_W"]) + beta[2]
               for r in probed}
        Xh = np.column_stack([np.array([hat[k] for k in cell]), X[:, 1]])
        lr = LogisticRegression(max_iter=2000).fit(Xh[tr], y[tr])
        scores[a] = roc_auc_score(y[te], lr.decision_function(Xh[te]))
    out.append(("pred D_out + M", scores))
    return out, X, y, grp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True,
                    help="result jsons, highest priority first; each pairs with "
                         "its <stem>_tokens/ directory if that exists")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows, tok = load(args.results)
    print(f"{len(rows)} (adapter, tau) cells; {len(tok)} with token probes")

    hdr = (f"\n{'adapter':11s} {'tau':>4} {'S':>7} {'L_W':>6} {'P':>7} "
           f"{'D_prompt':>9} {'D_out':>9} {'ratio':>7} {'E p90':>6} {'M p10':>6} {'T':>6} "
           f"{'tokflip':>8} {'seqflip':>8} {'churn':>7} {'net dU':>8}")
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["adapter"], -TAU_ORDER.get(r["tau"], 0))):
        t = next((t for t in tok if t["adapter"] == r["adapter"] and t["tau"] == r["tau"]), None)
        head = (f"{r['adapter']:11s} {r['tau']:>4} {r['S']:7.4f} {r['L_W']:6.3f} {r['P']:7.4f} "
                f"{r['d_prompt']:9.5f}")
        if t is None:
            print(head + " " + " ".join(f"{'-':>{w}}" for w in (9, 7, 6, 6, 6, 8, 8))
                  + f" {r['churn']:7.1%} {r['net']:+8.1%}")
            continue
        M, Mc, J = t["M"], t["Mc"], t["J"]
        E, flip = M - Mc, Mc < 0
        ex = np.unique(t["ex"])
        T = t["ex"].size / max(ex.size, 1)
        seq = np.unique(t["ex"][flip]).size / max(ex.size, 1)
        print(head + f" {r['d_out']:9.5f} {r['d_prompt'] / max(r['d_out'], 1e-12):7.2f} "
              f"{np.percentile(E, 90):6.2f} {np.percentile(M, 10):6.2f} {T:6.1f} "
              f"{flip.mean():8.2%} {seq:8.1%} {r['churn']:7.1%} {r['net']:+8.1%}")
    fit_law(rows)

    a = fig_perturbation_vs_utility(rows, args.outdir / "rq3_perturbation_vs_utility.pdf")
    print(f"\nwrote {a}")
    if not tok:
        print("no token probes found -- rerun predibase_task_metrics.py for figs 2 and 3")
        return

    b, grid, cnt, qj, qm = fig_flip_grid(tok, args.outdir / "rq3_flip_by_js_margin.pdf")
    print(f"wrote {b}")
    print("\nP(flip) by J (rows, high->low) x M (cols, low->high)")
    print("  J quartile edges: " + ", ".join(f"{v:.2e}" for v in qj[1:-1]))
    print("  M quartile edges: " + ", ".join(f"{v:.3f}" for v in qm[1:-1]))
    for i, row in enumerate(grid):
        print("  " + " ".join(f"{v:7.1%}" if np.isfinite(v) else "      -" for v in row))

    c = fig_erosion_vs_margin(tok, args.outdir / "rq3_erosion_vs_margin.pdf")
    print(f"wrote {c}")

    res, X, y, grp = logistic_check(tok, rows)
    print(f"\nleave-one-adapter-out AUROC for P(flip)   (n={len(y):,} tokens, "
          f"{y.mean():.1%} positive)")
    names = sorted(set(grp))
    print(f"  {'model':14s} " + " ".join(f"{a[:9]:>10s}" for a in names) + f" {'mean':>7}")
    for name, sc in res:
        vals = [sc.get(a) for a in names]
        mean = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else float("nan")
        print(f"  {name:14s} " + " ".join(f"{v:10.3f}" if v is not None else f"{'-':>10}"
                                          for v in vals) + f" {mean:7.3f}")


if __name__ == "__main__":
    main()
