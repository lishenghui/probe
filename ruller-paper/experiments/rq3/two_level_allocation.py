#!/usr/bin/env python3
"""Core routines for sensitivity-aware, two-level dense rank allocation."""
from __future__ import annotations

import heapq
import math
from collections.abc import Sequence


def layer_loss(singular: Sequence[float], rank: int) -> float:
    """Relative Frobenius residual of one module at ``rank``."""
    energy = [float(x) ** 2 for x in singular]
    total = sum(energy)
    if total == 0:
        return 0.0
    return math.sqrt(max(0.0, sum(energy[rank:]) / total))


def allocate_ranks(
    sigma: Sequence[Sequence[float]],
    sensitivity: Sequence[float],
    budget: int,
    gamma: float = 1.0,
    min_rank: int = 1,
) -> tuple[list[int], float]:
    """Greedily maximize the requested marginal weighted-risk reduction.

    The priority is exactly

        S_l**gamma * (L_l(k) - L_l(k + 1)).

    This intentionally differs from sorting weighted singular-value energy:
    the objective in the method is a sum of *relative residual norms*, not a
    sum of squared residuals.
    """
    if len(sigma) != len(sensitivity):
        raise ValueError("sigma/sensitivity length mismatch")
    if min_rank < 0:
        raise ValueError("min_rank must be non-negative")
    capacities = [len(s) for s in sigma]
    floor = sum(min(min_rank, n) for n in capacities)
    ceiling = sum(capacities)
    if not floor <= budget <= ceiling:
        raise ValueError(f"budget {budget} outside feasible [{floor}, {ceiling}]")

    ranks = [min(min_rank, n) for n in capacities]
    losses = [[layer_loss(s, k) for k in range(len(s) + 1)] for s in sigma]
    heap: list[tuple[float, int]] = []

    def push(module: int) -> None:
        k = ranks[module]
        if k < capacities[module]:
            gain = max(float(sensitivity[module]), 0.0) ** gamma * (
                losses[module][k] - losses[module][k + 1]
            )
            heapq.heappush(heap, (-gain, module))

    for module in range(len(sigma)):
        push(module)
    for _ in range(budget - floor):
        if not heap:
            raise RuntimeError("rank heap exhausted before reaching budget")
        _, module = heapq.heappop(heap)
        ranks[module] += 1
        push(module)

    risk = sum(max(float(s), 0.0) ** gamma * losses[m][ranks[m]]
               for m, s in enumerate(sensitivity))
    return ranks, risk


def allocate_squared_ranks(
    sigma: Sequence[Sequence[float]],
    sensitivity: Sequence[float],
    budget: int,
    gamma: float = 2.0,
    min_rank: int = 1,
) -> tuple[list[int], float]:
    """Exact top-K allocation for normalized squared residual risk.

    Minimizes ``sum_l S_l**gamma * residual_energy_l / total_energy_l``.
    Within a layer the weighted normalized singular energies are decreasing, so
    global top-K selection automatically respects every layer's rank prefix.
    """
    if len(sigma) != len(sensitivity):
        raise ValueError("sigma/sensitivity length mismatch")
    if min_rank < 0:
        raise ValueError("min_rank must be non-negative")
    capacities = [len(s) for s in sigma]
    ranks = [min(min_rank, n) for n in capacities]
    floor, ceiling = sum(ranks), sum(capacities)
    if not floor <= budget <= ceiling:
        raise ValueError(f"budget {budget} outside feasible [{floor}, {ceiling}]")

    totals = [sum(float(x) ** 2 for x in s) for s in sigma]
    directions = []
    for module, singular in enumerate(sigma):
        weight = max(float(sensitivity[module]), 0.0) ** gamma
        denom = totals[module]
        for index in range(ranks[module], len(singular)):
            utility = (0.0 if denom == 0 else
                       weight * float(singular[index]) ** 2 / denom)
            directions.append((utility, module, index))
    directions.sort(reverse=True)
    for _, module, _ in directions[:budget - floor]:
        ranks[module] += 1

    risk = 0.0
    for module, singular in enumerate(sigma):
        if totals[module] > 0:
            residual = sum(float(x) ** 2 for x in singular[ranks[module]:])
            risk += max(float(sensitivity[module]), 0.0) ** gamma * residual / totals[module]
    return ranks, risk


def allocate_global_spectral_ranks(
    sigma: Sequence[Sequence[float]],
    sensitivity: Sequence[float],
    budget: int,
    gamma: float = 0.0,
    min_rank: int = 1,
) -> tuple[list[int], float]:
    """Exact original A-SCT inner candidate at a fixed total rank.

    Minimizes the adapter-wide squared Frobenius residual, so optional
    directions are ordered by raw singular energy (not layer-normalized
    energy). ``sensitivity`` and ``gamma`` are accepted for allocator API
    compatibility and intentionally ignored.
    """
    del sensitivity, gamma
    capacities = [len(s) for s in sigma]
    ranks = [min(min_rank, n) for n in capacities]
    floor, ceiling = sum(ranks), sum(capacities)
    if not floor <= budget <= ceiling:
        raise ValueError(f"budget {budget} outside feasible [{floor}, {ceiling}]")
    directions = [(float(value) ** 2, module, index)
                  for module, singular in enumerate(sigma)
                  for index, value in enumerate(singular[ranks[module]:], ranks[module])]
    directions.sort(reverse=True)
    for _, module, _ in directions[:budget - floor]:
        ranks[module] += 1
    total = sum(float(x) ** 2 for singular in sigma for x in singular)
    residual = sum(float(x) ** 2 for module, singular in enumerate(sigma)
                   for x in singular[ranks[module]:])
    return ranks, (0.0 if total == 0 else residual / total)


def allocate_functional_dp(
    costs: Sequence[Sequence[float]],
    budget: int,
    min_rank: int = 1,
) -> tuple[list[int], float]:
    """Exact multiple-choice knapsack for additive isolated-layer E2E costs.

    ``costs[l][k]`` is the measured end-to-end damage when only module ``l``
    is truncated to rank ``k`` and every other module remains full rank.  The
    returned allocation exactly minimizes ``sum_l costs[l][k_l]`` subject to
    ``sum_l k_l == budget``.
    """
    if min_rank < 0:
        raise ValueError("min_rank must be non-negative")
    capacities = [len(row) - 1 for row in costs]
    if any(n < min_rank for n in capacities):
        raise ValueError("cost table does not cover the minimum rank")
    floor, ceiling = len(costs) * min_rank, sum(capacities)
    if not floor <= budget <= ceiling:
        raise ValueError(f"budget {budget} outside feasible [{floor}, {ceiling}]")

    inf = float("inf")
    dp = [inf] * (budget + 1)
    dp[0] = 0.0
    parents: list[list[tuple[int, int] | None]] = []
    for row, capacity in zip(costs, capacities):
        nxt = [inf] * (budget + 1)
        parent: list[tuple[int, int] | None] = [None] * (budget + 1)
        for used, value in enumerate(dp):
            if not math.isfinite(value):
                continue
            for rank in range(min_rank, capacity + 1):
                new_used = used + rank
                if new_used > budget:
                    break
                candidate = value + float(row[rank])
                if candidate < nxt[new_used]:
                    nxt[new_used] = candidate
                    parent[new_used] = (used, rank)
        dp = nxt
        parents.append(parent)
    if not math.isfinite(dp[budget]):
        raise RuntimeError("functional DP could not reach exact budget")
    ranks = [0] * len(costs)
    used = budget
    for module in range(len(costs) - 1, -1, -1):
        choice = parents[module][used]
        if choice is None:
            raise RuntimeError("broken functional DP parent chain")
        used, ranks[module] = choice
    return ranks, dp[budget]


def minimax_allocate(curves: dict[str, list[dict]], budget: int) -> tuple[dict, float, int]:
    """Exact discrete minimax allocation over measured adapter curves.

    For any damage ceiling, the cheapest feasible point of every adapter is
    sufficient. Searching the finite set of measured damages is therefore exact.
    Remaining slack is spent by the largest measured one-step damage reduction.
    """
    if not curves:
        raise ValueError("no curves")
    clean = {name: sorted(rows, key=lambda r: int(r["k"]))
             for name, rows in curves.items()}
    breakpoints = sorted({float(r["d_js"]) for rows in clean.values() for r in rows})

    def under(ceiling: float):
        chosen = {}
        for name, rows in clean.items():
            feasible = [r for r in rows if float(r["d_js"]) <= ceiling]
            if not feasible:
                return None
            chosen[name] = min(feasible, key=lambda r: int(r["k"]))
        return chosen

    lo, hi, best = 0, len(breakpoints) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        chosen = under(breakpoints[mid])
        spent = math.inf if chosen is None else sum(int(r["k"]) for r in chosen.values())
        if spent <= budget:
            best = (breakpoints[mid], chosen)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        raise ValueError("budget infeasible on measured curves")

    ceiling, _ = best
    # Secondary exact objective: among allocations attaining the optimal
    # minimax ceiling, minimize total measured damage.  A one-step greedy pass
    # is insufficient because measured curves are noisy/non-monotone: reaching
    # a much better point can require crossing a worse intermediate point.
    # Ties prefer greater budget utilization.
    states = {0: (0.0, {})}
    for name, rows in clean.items():
        feasible = [r for r in rows if float(r["d_js"]) <= ceiling]
        nxt_states = {}
        for used, (total_damage, allocation) in states.items():
            for row in feasible:
                new_used = used + int(row["k"])
                if new_used > budget:
                    continue
                candidate = (total_damage + float(row["d_js"]),
                             {**allocation, name: row})
                incumbent = nxt_states.get(new_used)
                if incumbent is None or candidate[0] < incumbent[0]:
                    nxt_states[new_used] = candidate
        states = nxt_states
    if not states:
        raise ValueError("secondary allocation infeasible")
    spent, (_, chosen) = min(states.items(), key=lambda item: (item[1][0], -item[0]))
    return chosen, ceiling, spent
