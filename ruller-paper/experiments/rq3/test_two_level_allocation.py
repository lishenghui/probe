import itertools

from two_level_allocation import (allocate_global_spectral_ranks, allocate_ranks,
                                  allocate_functional_dp, allocate_squared_ranks,
                                  minimax_allocate)


def test_sensitivity_changes_layer_choice():
    sigma = [[4, 3, 1], [4, 3, 1]]
    ranks, _ = allocate_ranks(sigma, [10, 1], budget=3)
    assert ranks == [2, 1]


def test_rank_floor_and_budget_are_exact():
    ranks, risk = allocate_ranks([[3, 2], [1]], [1, 1], budget=3)
    assert ranks == [2, 1]
    assert risk == 0


def test_measured_minimax():
    curves = {
        "a": [{"k": 1, "d_js": .8}, {"k": 2, "d_js": .1}],
        "b": [{"k": 1, "d_js": .6}, {"k": 2, "d_js": .2}],
    }
    chosen, ceiling, spent = minimax_allocate(curves, 3)
    assert spent == 3
    assert chosen["a"]["k"] == 2
    assert chosen["b"]["k"] == 1
    assert ceiling == .6


def test_minimax_secondary_objective_crosses_noisy_barrier():
    curves = {
        "a": [{"k": 1, "d_js": .5}, {"k": 2, "d_js": .6},
              {"k": 3, "d_js": .1}],
        "b": [{"k": 1, "d_js": .5}, {"k": 2, "d_js": .4}],
    }
    chosen, ceiling, spent = minimax_allocate(curves, 4)
    assert ceiling == .5
    assert spent == 4
    assert chosen["a"]["k"] == 3
    assert chosen["b"]["k"] == 1


def test_squared_topk_matches_bruteforce():
    sigma = [[5, 3, 1], [4, 2], [7, 1, .5]]
    sensitivity = [0.3, 2.0, 0.8]
    budget, gamma = 5, 2.0
    ranks, risk = allocate_squared_ranks(sigma, sensitivity, budget, gamma)

    def objective(candidate):
        total = 0.0
        for singular, strength, rank in zip(sigma, sensitivity, candidate):
            den = sum(x * x for x in singular)
            total += strength ** gamma * sum(x * x for x in singular[rank:]) / den
        return total

    candidates = [r for r in itertools.product(*(range(1, len(s) + 1) for s in sigma))
                  if sum(r) == budget]
    assert abs(risk - min(map(objective, candidates))) < 1e-12
    assert abs(objective(ranks) - risk) < 1e-12


def test_global_spectral_uses_raw_not_layer_normalized_energy():
    sigma = [[100, 9], [2, 1.9]]
    ranks, _ = allocate_global_spectral_ranks(sigma, [1, 1], budget=3)
    assert ranks == [2, 1]
    normalized, _ = allocate_squared_ranks(sigma, [1, 1], budget=3, gamma=0)
    assert normalized == [1, 2]


def test_functional_dp_matches_bruteforce_with_nonmonotone_costs():
    costs = [[0, .8, .3, 0], [0, .4, .5, 0], [0, .9, .2, 0]]
    budget = 6
    ranks, risk = allocate_functional_dp(costs, budget)
    candidates = [r for r in itertools.product(range(1, 4), repeat=3)
                  if sum(r) == budget]
    objective = lambda r: sum(costs[i][k] for i, k in enumerate(r))
    assert abs(risk - min(map(objective, candidates))) < 1e-12
    assert abs(objective(ranks) - risk) < 1e-12
