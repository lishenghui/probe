#!/usr/bin/env python3
"""Build the compact Uniform/SCT/A-SCT/Oracle main-result table."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

TAUS = [.99, .95, .90, .80, .70, .50]

def load_rows(pattern):
    out = {}
    for path in glob.glob(pattern):
        for row in json.load(open(path)):
            name = row.get("short") or row.get("adapter")
            dst = out.setdefault(name, {k:v for k,v in row.items() if k != "variants"})
            dst.setdefault("variants", {}).update(row["variants"])
    return out

def grid_rows(pattern, nominal, min_headroom=.05):
    raw, out = load_rows(pattern), {}
    for name, row in raw.items():
        floor = row.get("metric_base")
        if floor is None:
            continue
        head = row.get("headroom", row["metric_orig"]-floor)
        if head < min_headroom or not all(f"e{round(t*100):02d}" in row["variants"] for t in TAUS):
            continue
        out[name] = []
        for tau in TAUS:
            v = row["variants"][f"e{round(tau*100):02d}"]
            u = v.get("retained")
            if u is None: u = (v["metric"]-floor)/head
            out[name].append(dict(tau=tau, k=round(v["rank_frac"]*nominal), u=float(u)))
    return out

def oracle(grid, budget):
    # Lexicographic labeled oracle: maximise the worst retention, then mean.
    levels = sorted({x["u"] for rows in grid.values() for x in rows}, reverse=True)
    floor = None
    for q in levels:
        need = 0
        for rows in grid.values():
            ok = [x["k"] for x in rows if x["u"] >= q]
            if not ok: need = budget+1; break
            need += min(ok)
        if need <= budget:
            floor = q; break
    names = sorted(grid); neg = -1e100
    dp = np.full(budget+1, neg); dp[0] = 0
    parents = []
    for name in names:
        nd = np.full(budget+1, neg)
        choice = np.full(budget+1, -1, dtype=int)
        prevk = np.full(budget+1, -1, dtype=int)
        for j,x in enumerate(grid[name]):
            if x["u"] + 1e-12 < floor: continue
            k=x["k"]; vals=dp[:-k]+x["u"]; better=vals>nd[k:]
            inds=np.flatnonzero(better)+k
            nd[inds]=vals[better]; choice[inds]=j; prevk[inds]=inds-k
        dp=nd;parents.append((choice,prevk))
    k=int(np.nanargmax(dp)); selected={}
    for i in range(len(names)-1,-1,-1):
        ch,pk=parents[i]; j=int(ch[k]); selected[names[i]]=grid[names[i]][j]; k=int(pk[k])
    return selected

def stats(rows):
    u=np.array([x["u"] for x in rows.values()])
    return dict(mean=float(u.mean()), p10=float(np.percentile(u,10)),
                worst=float(u.min()), broken=int((u<=0).sum()),
                spent=sum(x["k"] for x in rows.values()))

def main():
    pools = [
      ("LoRA Land",12,755,
       grid_rows("artifacts/rq3/results/land12_task_grid.json",512),
       "artifacts/rq3/results/grid_alloc_land12_original.json",
       "artifacts/rq3/results/grid_alloc_land12_anchor2.json", "2$\\times$32 unlabeled"),
      ("Lots-of-LoRAs",19,6401,
       grid_rows("artifacts/rq3/results/cts_task*.json",1536),
       "artifacts/rq3/results/grid_alloc_cts_original_full.json",
       "artifacts/rq3/results/grid_alloc_cts_disjoint_anchor2.json", "2$\\times$48 unlabeled"),
      ("LoRARetriever",41,3191,
       grid_rows("artifacts/rq3/results/lorare_task_disjoint_s*.json",512),
       "artifacts/rq3/results/grid_alloc_lorare_disjoint_original.json",
       "artifacts/rq3/results/grid_alloc_lorare_disjoint_anchor2.json", "2$\\times$16 unlabeled"),
    ]
    # The allocation files already carry every budget the sweep visited, so the
    # loose operating point costs nothing extra to report. tau = .50 is omitted:
    # it is the coarsest evaluated level, the budget it defines equals the sum of
    # every adapter's coarsest option, and the allocator therefore has exactly one
    # feasible solution -- all four rules including Oracle return identical
    # allocations, which says something about the evaluation grid and nothing
    # about the rules.
    WANT_TAU = [.90, .70]
    result={}
    for name,n,_,grid,sctf,asctf,cal in pools:
        S=json.load(open(sctf))["budgets"]; A=json.load(open(asctf))["budgets"]
        by_tau={round(v["tau"],2):k for k,v in A.items()}
        result[name]=dict(n=n,calibration=cal,budgets={})
        for tau in WANT_TAU:
            key=by_tau.get(round(tau,2))
            if key is None:
                print(f"  {name}: no allocation stored at tau={tau}"); continue
            sct,asc=S[key],A[key]
            uni={k:v["uniform"] for k,v in asc["paired"].items()}
            sr={k:v["sct"] for k,v in sct["paired"].items()}
            ar={k:v["sct"] for k,v in asc["paired"].items()}
            # The grid comes from a glob and the allocations from stored files, so
            # topping a pool up leaves the Oracle wider than the rules it is
            # compared against. Score every rule on the same members.
            members=sorted(asc["paired"])
            missing=[m for m in members if m not in grid]
            if missing:
                raise SystemExit(f"{name}: no measured grid for {missing}")
            ora=oracle({m:grid[m] for m in members},int(key))
            result[name]["budgets"][key]=dict(
                tau=tau, methods={"Uniform":stats(uni),"SCT":stats(sr),
                                  "A-SCT":stats(ar),"Oracle":stats(ora)})
    out=Path("artifacts/rq3/results/main_allocation_table.json")
    out.write_text(json.dumps(result,indent=2)+"\n")
    for name,p in result.items():
        print(f"\n{name} (N={p['n']}, {p['calibration']})")
        for key,b in p["budgets"].items():
            print(f"  budget {key} (uniform tau={b['tau']})")
            for method,s in b["methods"].items():
                print(f"    {method:8s} worst={s['worst']:7.3f} p10={s['p10']:7.3f} "
                      f"mean={s['mean']:6.3f} broken={s['broken']} spent={s['spent']}")
    print("wrote",out)

if __name__ == "__main__": main()
