#!/usr/bin/env python3
"""Exact labeled oracle over a finite union of per-adapter candidates.

Input schema::

  {"budget": 100, "adapters": {
    "task_a": [{"k": 20, "u": .9, "source": "FRA-0"}, ...], ...}}

The solver lexicographically maximizes (worst, P10, mean) subject to the
fleet-rank budget.  P10 is exactly NumPy's ``quantile(..., .1,
method='linear')``.  This is an exact Pareto dynamic program, not a greedy
approximation.  It is intended for the small candidate union in Table 1.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class State:
    cost: int
    bottom: tuple[float, ...]
    total: float
    choices: tuple[int, ...]


def _linear_quantile_from_bottom(bottom: tuple[float, ...], n: int,
                                 q: float = 0.1) -> float:
    h = (n - 1) * q
    lo = int(math.floor(h))
    frac = h - lo
    return (1.0 - frac) * bottom[lo] + frac * bottom[lo + 1]


def _insert_bottom(bottom: tuple[float, ...], value: float,
                   keep: int) -> tuple[float, ...]:
    return tuple(sorted((*bottom, value))[:keep])


def _dominates(a: State, b: State) -> bool:
    """Safe dominance under every possible common future continuation."""
    return (a.cost <= b.cost and a.total >= b.total
            and all(x >= y for x, y in zip(a.bottom, b.bottom)))


def _pareto(states: list[State]) -> list[State]:
    # Exact duplicates in objective state only need the first reconstruction.
    unique: dict[tuple[int, tuple[float, ...]], State] = {}
    for state in states:
        key = (state.cost, state.bottom)
        old = unique.get(key)
        if old is None or state.total > old.total:
            unique[key] = state
    ordered = sorted(unique.values(), key=lambda s: (s.cost, -s.total))
    frontier: list[State] = []
    for state in ordered:
        if any(_dominates(old, state) for old in frontier):
            continue
        frontier = [old for old in frontier if not _dominates(state, old)]
        frontier.append(state)
    return frontier


def _validate_and_prune(adapters: dict[str, list[dict[str, Any]]],
                        budget: int) -> tuple[list[str], list[list[dict[str, Any]]]]:
    if not adapters:
        raise ValueError("adapters must be non-empty")
    names = list(adapters)
    result: list[list[dict[str, Any]]] = []
    for name in names:
        raw = adapters[name]
        if not raw:
            raise ValueError(f"{name}: empty candidate list")
        candidates = []
        for c in raw:
            k, u = int(c["k"]), float(c["u"])
            if k < 0 or not math.isfinite(u):
                raise ValueError(f"{name}: invalid candidate {c}")
            if k <= budget:
                candidates.append({**c, "k": k, "u": u,
                                   "source": str(c.get("source", "unknown"))})
        # A cheaper candidate with no lower utility universally dominates.
        kept = [c for i, c in enumerate(candidates)
                if not any(j != i and d["k"] <= c["k"] and d["u"] >= c["u"]
                           and (d["k"] < c["k"] or d["u"] > c["u"])
                           for j, d in enumerate(candidates))]
        if not kept:
            raise ValueError(f"{name}: no candidate fits budget {budget}")
        result.append(kept)
    if sum(min(c["k"] for c in cs) for cs in result) > budget:
        raise ValueError("budget is below the sum of per-adapter minimum costs")
    return names, result


def solve(adapters: dict[str, list[dict[str, Any]]], budget: int,
          max_states: int = 1_000_000) -> dict[str, Any]:
    """Return the exact lexicographic candidate-union oracle allocation."""
    names, candidates = _validate_and_prune(adapters, budget)
    n = len(names)
    h = (n - 1) * 0.1
    keep = min(n, int(math.floor(h)) + 2)
    states = [State(0, (), 0.0, ())]
    frontier_sizes = []
    for cs in candidates:
        expanded = [State(s.cost + c["k"], _insert_bottom(s.bottom, c["u"], keep),
                          s.total + c["u"], (*s.choices, ci))
                    for s in states for ci, c in enumerate(cs)
                    if s.cost + c["k"] <= budget]
        states = _pareto(expanded)
        frontier_sizes.append(len(states))
        if len(states) > max_states:
            raise RuntimeError(f"exact frontier has {len(states)} states; "
                               f"increase --max-states (no approximate result emitted)")
    if not states:
        raise RuntimeError("no feasible allocation")

    def objective(s: State) -> tuple[float, float, float, int]:
        worst = s.bottom[0]
        p10 = _linear_quantile_from_bottom(s.bottom, n)
        return worst, p10, s.total / n, -s.cost

    best = max(states, key=objective)
    allocation = {name: candidates[i][ci]
                  for i, (name, ci) in enumerate(zip(names, best.choices))}
    utilities = np.asarray([allocation[name]["u"] for name in names], dtype=float)
    stats = {"mean": float(utilities.mean()),
             "p10": float(np.quantile(utilities, 0.1, method="linear")),
             "worst": float(utilities.min())}
    return {"budget": budget, "spent": best.cost,
            "objective_order": ["worst", "p10", "mean"],
            "quantile": {"q": 0.1, "method": "linear"},
            "stats": stats, "allocation": allocation,
            "solver": {"type": "exact_pareto_dp", "frontier_sizes": frontier_sizes,
                       "final_states": len(states)}}


def _self_test() -> None:
    rng = np.random.default_rng(7)
    for n in range(2, 8):
        for _ in range(20):
            adapters = {f"a{i}": [{"k": int(rng.integers(0, 5)),
                                     "u": float(rng.integers(0, 11)) / 10,
                                     "source": f"m{j}"} for j in range(3)]
                        for i in range(n)}
            budget = int(rng.integers(4, 15))
            if sum(min(c["k"] for c in cs) for cs in adapters.values()) > budget:
                continue
            got = solve(adapters, budget)
            feasible = (xs for xs in itertools.product(*adapters.values())
                        if sum(x["k"] for x in xs) <= budget)
            def key(xs: tuple[dict[str, Any], ...]) -> tuple[float, float, float, int]:
                us = np.asarray([x["u"] for x in xs])
                cost = sum(x["k"] for x in xs)
                return float(us.min()), float(np.quantile(us, .1)), float(us.mean()), -cost
            expected = max(feasible, key=key)
            assert np.allclose(tuple(got["stats"][x] for x in ("worst", "p10", "mean")),
                               key(expected)[:3])
    print("self-test passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, nargs="?")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--budget", type=int, help="override input budget")
    ap.add_argument("--max-states", type=int, default=1_000_000)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.input is None:
        ap.error("input is required unless --self-test is used")
    doc = json.loads(args.input.read_text())
    budget = args.budget if args.budget is not None else int(doc["budget"])
    result = solve(doc["adapters"], budget, args.max_states)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
