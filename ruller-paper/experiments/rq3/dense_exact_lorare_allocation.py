#!/usr/bin/env python3
"""Dense exact A-SCT allocation for LoRARetriever.

Build every attainable adapter-level retained-rank breakpoint from the saved
per-module spectra, predict its anchored damage, and solve the discrete minimax
problem exactly.  No downstream labels are used by the allocator.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


def load_rows(patterns: list[str]) -> dict:
    out = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            for row in json.loads(Path(path).read_text()):
                name = row.get("short") or row.get("adapter")
                dst = out.setdefault(name, {"variants": {}})
                dst.update({k: v for k, v in row.items() if k != "variants"})
                dst["variants"].update(row["variants"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", type=Path, required=True)
    ap.add_argument("--anchor", nargs="+", required=True)
    ap.add_argument("--task", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=3191)
    ap.add_argument("--anchor-tau", type=float, default=.95)
    ap.add_argument("--fixed-b", type=float, default=3.7834836286870286)
    ap.add_argument("--min-headroom", type=float, default=.05)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    spectra = json.loads(args.spectra.read_text())
    anchor, task = load_rows(args.anchor), load_rows(args.task)
    label = f"e{round(args.anchor_tau * 100):02d}"
    names = sorted(n for n, row in task.items()
                   if n in spectra and n in anchor and label in anchor[n]["variants"]
                   and row.get("headroom", row.get("metric_orig", 0) -
                               row.get("metric_base", 0)) >= args.min_headroom)

    curves = {}
    for name in names:
        modules = spectra[name]["sigma"]
        # Keeping globally largest remaining singular directions is the exact
        # minimum-Frobenius-loss allocation at every total adapter rank.  At
        # least one direction per module is retained for a realizable LoRA arm.
        mandatory = [float(s[0]) ** 2 for s in modules]
        optional = sorted(((float(x) ** 2, m) for m, s in enumerate(modules)
                           for x in s[1:]), reverse=True)
        total = sum(float(x) ** 2 for s in modules for x in s)
        kept, k0 = sum(mandatory), len(modules)
        av = anchor[name]["variants"][label]
        anchor_d, anchor_l = av["d_out"], av["L_W"]
        rows = []
        values = [kept]
        module_ranks = [1] * len(modules)
        rank_patterns = [list(module_ranks)]
        for energy, module in optional:
            values.append(values[-1] + energy)
            module_ranks[module] += 1
            rank_patterns.append(list(module_ranks))
        for j, energy in enumerate(values):
            k = k0 + j
            loss = math.sqrt(max(0.0, 1.0 - energy / total))
            damage = (0.0 if loss == 0 else
                      anchor_d * (loss / anchor_l) ** args.fixed_b)
            rows.append({"k": k, "L_W": loss, "predicted_D": damage,
                         "module_ranks": rank_patterns[j]})
        curves[name] = rows

    breakpoints = sorted({r["predicted_D"] for rows in curves.values() for r in rows})

    def allocation(q: float):
        out = {}
        for name, rows in curves.items():
            candidates = [r for r in rows if r["predicted_D"] <= q]
            if not candidates:
                return None
            out[name] = min(candidates, key=lambda r: r["k"])
        return out

    lo, hi = 0, len(breakpoints) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        chosen = allocation(breakpoints[mid])
        spent = math.inf if chosen is None else sum(r["k"] for r in chosen.values())
        if spent <= args.budget:
            best = (breakpoints[mid], chosen)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        raise SystemExit("budget is infeasible even at the largest damage ceiling")
    q, chosen = best
    spent = sum(r["k"] for r in chosen.values())
    # The primary minimax optimum is exact. Spend slack only as a deterministic
    # tie-breaker, always upgrading the currently most damaged adapter.
    while spent < args.budget:
        upgrades = []
        for name, row in chosen.items():
            nxt = next((r for r in curves[name] if r["k"] == row["k"] + 1), None)
            if nxt is not None:
                upgrades.append((row["predicted_D"] - nxt["predicted_D"], name, nxt))
        if not upgrades:
            break
        _, name, nxt = max(upgrades)
        chosen[name] = nxt
        spent += 1

    result = {"pool": "LoRARetriever", "n": len(names), "budget": args.budget,
              "anchor_tau": args.anchor_tau, "fixed_b": args.fixed_b,
              "optimal_ceiling": q, "spent": spent, "allocation": chosen}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"dense exact: n={len(names)} spent={spent}/{args.budget} q*={q:.6e}")
    print(f"rank range: {min(r['k'] for r in chosen.values())}--"
          f"{max(r['k'] for r in chosen.values())}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
