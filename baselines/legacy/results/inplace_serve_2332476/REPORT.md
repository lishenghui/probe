# Paired dynamic serving optimization

GPU: NVIDIA GH200 120GB. Fixed observation, 16 concurrent clients, same banks across modes. Legacy compressed banks are performance-only.

Selected split-K: 1; calibration: fixed to previous serving winner for paired comparison

| Family | Adapters | Max batch | Mode | Runs | Median req/s | Range req/s | Median p95 ms |
|---|---:|---:|---|---:|---:|---|---:|
| compressed | 4 | 8 | heterogeneous | 2 | 23.50 | 23.39–23.60 | 736.5 |
| compressed | 4 | 8 | heterogeneous_inplace | 2 | 24.00 | 23.77–24.22 | 715.5 |
| compressed | 8 | 8 | heterogeneous | 2 | 23.54 | 23.53–23.55 | 741.9 |
| compressed | 8 | 8 | heterogeneous_inplace | 2 | 23.91 | 23.55–24.28 | 737.5 |
| compressed | 16 | 8 | base_only_probe | 1 | 31.51 | 31.51–31.51 | 561.3 |
| compressed | 16 | 8 | heterogeneous | 2 | 23.08 | 23.08–23.08 | 753.9 |
| compressed | 16 | 8 | heterogeneous_inplace | 2 | 23.14 | 22.67–23.61 | 754.3 |
| compressed | 16 | 16 | heterogeneous | 2 | 29.11 | 28.87–29.35 | 564.9 |
| compressed | 16 | 16 | heterogeneous_inplace | 2 | 28.86 | 28.67–29.05 | 586.6 |
| mixed_high_low | 8 | 8 | heterogeneous | 2 | 22.82 | 22.48–23.16 | 739.2 |
| mixed_high_low | 8 | 8 | heterogeneous_inplace | 2 | 23.42 | 23.25–23.59 | 728.3 |
| verified_r64 | 4 | 8 | heterogeneous | 2 | 22.24 | 22.21–22.27 | 771.0 |
| verified_r64 | 4 | 8 | heterogeneous_inplace | 2 | 22.95 | 22.30–23.60 | 743.0 |

## Same batch kernel comparisons

- compressed, 4 adapters, batch cap 8: 1.021x throughput (heterogeneous_inplace versus heterogeneous).
- compressed, 8 adapters, batch cap 8: 1.016x throughput (heterogeneous_inplace versus heterogeneous).
- compressed, 16 adapters, batch cap 8: 1.003x throughput (heterogeneous_inplace versus heterogeneous).
- compressed, 16 adapters, batch cap 16: 0.991x throughput (heterogeneous_inplace versus heterogeneous).
- mixed_high_low, 8 adapters, batch cap 8: 1.026x throughput (heterogeneous_inplace versus heterogeneous).
- verified_r64, 4 adapters, batch cap 8: 1.032x throughput (heterogeneous_inplace versus heterogeneous).

## Numerical parity

Maximum same-batch action difference: 0.00000000. This is fixed-input parity, not rollout validation.

## Profiling (separate instrumented runs)

### heterogeneous

| Region | CPU total ms | GPU total ms | Calls |
|---|---:|---:|---:|
| fleet.base | 0.00 | 106.02 | 640 |
| fleet.lora | 0.00 | 43.29 | 621 |

### heterogeneous_inplace

| Region | CPU total ms | GPU total ms | Calls |
|---|---:|---:|---:|
| fleet.base | 0.00 | 105.63 | 640 |
| fleet.lora | 0.00 | 39.62 | 621 |

### base_only_probe

| Region | CPU total ms | GPU total ms | Calls |
|---|---:|---:|---:|
| fleet.base | 0.00 | 105.20 | 640 |


## New JIT variants during measured requests

| Mode | Adapter count | Batch cap | Repeat | New variants |
|---|---:|---:|---:|---:|
| heterogeneous_inplace | 8 | 8 | 1 | 0 |
| heterogeneous | 16 | 16 | 1 | 0 |
| heterogeneous | 16 | 8 | 1 | 0 |
| heterogeneous_inplace | 8 | 8 | 0 | 0 |
| heterogeneous_inplace | 8 | 8 | 1 | 0 |
| heterogeneous | 8 | 8 | 0 | 0 |
| heterogeneous | 8 | 8 | 1 | 0 |
| heterogeneous | 4 | 8 | 0 | 0 |
| heterogeneous_inplace | 16 | 16 | 1 | 0 |
| heterogeneous_inplace | 4 | 8 | 0 | 0 |
| heterogeneous | 8 | 8 | 1 | 0 |
| heterogeneous | 4 | 8 | 0 | 0 |
| heterogeneous_inplace | 4 | 8 | 0 | 0 |
| heterogeneous | 8 | 8 | 0 | 0 |
| heterogeneous_inplace | 16 | 16 | 0 | 0 |
| heterogeneous | 16 | 16 | 0 | 0 |
| heterogeneous | 4 | 8 | 1 | 0 |
| heterogeneous | 4 | 8 | 1 | 0 |
| heterogeneous_inplace | 4 | 8 | 1 | 0 |
| heterogeneous_inplace | 4 | 8 | 1 | 0 |
| heterogeneous_inplace | 8 | 8 | 0 | 0 |
| base_only_probe | 16 | 8 | 0 | 0 |
| heterogeneous_inplace | 16 | 8 | 0 | 0 |
| heterogeneous | 16 | 8 | 0 | 0 |
| heterogeneous_inplace | 16 | 8 | 1 | 0 |

Only device-only region records are displayed above; CPU-associated records of the same name are excluded to avoid double-counting. Profiling totals can include nested operations; they are not added across all profiler events. The base-only probe is a performance ceiling measurement, not a valid policy. Two repetitions provide an initial range, not a confidence interval. HTTP/network overhead is excluded.
