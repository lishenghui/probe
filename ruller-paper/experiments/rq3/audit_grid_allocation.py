#!/usr/bin/env python3
"""Audit whether grid allocation failures come from fitting or the surrogate."""
import glob, json, math
import numpy as np
from scipy.stats import spearmanr

TAUS = [.99, .95, .90, .80, .70, .50]

def load(patterns):
    out = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            for row in json.load(open(path)):
                name = row.get("short") or row.get("adapter")
                dst = out.setdefault(name, {})
                for key, value in row.items():
                    if key == "variants":
                        dst.setdefault(key, {}).update(value)
                    elif key not in dst or dst[key] is None:
                        dst[key] = value
    return out

def fit(cells):
    X = np.array([[1, math.log(s), math.log(l)] for s, l, _ in cells])
    y = np.log([d for _, _, d in cells])
    return np.linalg.lstsq(X, y, rcond=None)[0]

def main():
    task = load(["artifacts/rq3/results/lorare_task_s*.json"])
    div = load(["artifacts/rq3/results/lorare_div_s*.json",
                "artifacts/rq3/results/lorare_div_low_s*.json"])
    names, grid = [], {}
    for name, row in task.items():
        headroom = row.get("headroom", row.get("metric_orig", 0)-row.get("metric_base", 0))
        if headroom < .05 or name not in div or not div[name].get("S"):
            continue
        if not all(f"e{round(t*100):02d}" in row["variants"] for t in TAUS):
            continue
        names.append(name)
        grid[name] = []
        for tau in TAUS:
            label = f"e{round(tau*100):02d}"
            tv, dv = row["variants"][label], div[name]["variants"][label]
            utility = tv.get("retained")
            if utility is None:
                utility = (tv["metric"]-row["metric_base"])/headroom
            grid[name].append(dict(tau=tau, k=round(tv["rank_frac"]*512),
                                   u=utility, L=tv["L_W"], d=dv["d_out"]))
    names.sort()
    cells = [(div[n]["S"], x["L"], x["d"]) for n in names for x in grid[n] if x["d"] > 0]
    global_beta = fit(cells)
    loo_beta = {n: fit([(div[m]["S"], x["L"], x["d"])
                        for m in names if m != n for x in grid[m] if x["d"] > 0])
                for n in names}
    budget = sum(next(x for x in grid[n] if x["tau"] == .70)["k"] for n in names)

    def capalloc(score):
        values = sorted({score(n, x) for n in names for x in grid[n]})
        for cap in values:
            alloc = {}
            for n in names:
                feasible = [x for x in grid[n] if score(n, x) <= cap]
                alloc[n] = min(feasible, key=lambda x:x["k"]) if feasible else max(
                    grid[n], key=lambda x:x["k"])
            if sum(x["k"] for x in alloc.values()) <= budget:
                return alloc
        raise RuntimeError("no feasible cap")

    def predicted(beta, n, x):
        c, a, b = beta
        return c+a*math.log(div[n]["S"])+b*math.log(x["L"])

    uniform = {n: next(x for x in grid[n] if x["tau"] == .70) for n in names}
    allocations = {
        "uniform": uniform,
        "LOAO-predicted": capalloc(lambda n,x: predicted(loo_beta[n],n,x)),
        "global-predicted": capalloc(lambda n,x: predicted(global_beta,n,x)),
        "oracle-measured-Dout": capalloc(lambda n,x: math.log(x["d"])),
    }
    # A fleet can cheaply measure one mild-compression anchor per adapter.  This
    # absorbs adapter/task-specific intercepts while retaining the shared L slope.
    b_shared = global_beta[2]
    def anchored(n, x, tau=.90):
        anchor = next(z for z in grid[n] if z["tau"] == tau)
        return math.log(anchor["d"])+b_shared*(math.log(x["L"])-math.log(anchor["L"]))
    allocations["e99-anchor-predicted"] = capalloc(lambda n,x: anchored(n,x,.99))
    allocations["e95-anchor-predicted"] = capalloc(lambda n,x: anchored(n,x,.95))
    allocations["e90-anchor-predicted"] = capalloc(anchored)
    print(f"n={len(names)} budget={budget} global beta={global_beta}")
    logd, loss = [], []
    within = []
    for n in names:
        logd.extend(math.log(x["d"]) for x in grid[n])
        loss.extend(1-x["u"] for x in grid[n])
        within.append(spearmanr([x["d"] for x in grid[n]],
                                [1-x["u"] for x in grid[n]]).statistic)
    print("pooled Spearman(log Dout, utility loss):", spearmanr(logd, loss))
    for tau in TAUS:
        xs = [math.log(div[n]["S"]) for n in names]
        ys = [math.log(next(x for x in grid[n] if x["tau"] == tau)["d"]) for n in names]
        print(f"cross-adapter tau={tau:.2f}: rho(logS,logD)={spearmanr(xs,ys).statistic:.3f}")
    valid = [x for x in within if not np.isnan(x)]
    print("within-adapter rho median", np.median(valid), "negative", sum(x < 0 for x in valid))
    residuals = {}
    for n in names:
        residuals[n] = np.mean([math.log(x["d"])-predicted(global_beta,n,x) for x in grid[n]])
    vals = np.array(list(residuals.values()))
    print(f"adapter mean log-residual std={vals.std():.3f}, range={vals.min():.3f}..{vals.max():.3f} "
          f"({math.exp(vals.max()-vals.min()):.1f}x multiplicative span)")
    for label, alloc in allocations.items():
        u = np.array([alloc[n]["u"] for n in names])
        d = np.array([alloc[n]["d"] for n in names])
        print(f"{label:22s} spent={sum(x['k'] for x in alloc.values()):4d} "
              f"mean={u.mean():.3f} worst={u.min():.3f} p10={np.percentile(u,10):.3f} "
              f"median={np.median(u):.3f} maxD={d.max():.3e} p90D={np.percentile(d,90):.3e}")

    # Exact multiple-choice knapsack: the best possible mean utility on this measured grid.
    neg = -1e30
    dp = np.full(budget+1, neg); dp[0] = 0
    for n in names:
        nxt = np.full(budget+1, neg)
        for x in grid[n]:
            nxt[x["k"]:] = np.maximum(nxt[x["k"]:], dp[:-x["k"]]+x["u"])
        dp = nxt
    print(f"oracle-utility-sum     spent={int(dp.argmax()):4d} mean={dp.max()/len(names):.3f}")

if __name__ == "__main__":
    main()
