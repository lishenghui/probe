# Paired dynamic serving optimization

GPU: NVIDIA GH200 120GB. Fixed observation, 16 concurrent clients, same banks across modes. Legacy compressed banks are performance-only.

Selected split-K: 1; calibration: fixed to previous serving winner for paired comparison

| Family | Adapters | Max batch | Mode | Runs | Median req/s | Range req/s | Median p95 ms |
|---|---:|---:|---|---:|---:|---|---:|
| compressed | 4 | 8 | heterogeneous | 2 | 22.32 | 21.64–23.00 | 781.9 |
| compressed | 4 | 8 | segmented | 2 | 22.32 | 22.14–22.49 | 773.7 |
| compressed | 8 | 8 | heterogeneous | 2 | 22.49 | 22.32–22.65 | 788.9 |
| compressed | 8 | 8 | segmented | 2 | 18.13 | 14.28–21.98 | 2030.0 |
| compressed | 16 | 8 | base_only_probe | 1 | 29.92 | 29.92–29.92 | 578.0 |
| compressed | 16 | 8 | heterogeneous | 2 | 22.42 | 22.14–22.70 | 792.7 |
| compressed | 16 | 8 | segmented | 2 | 20.57 | 20.00–21.13 | 1026.5 |
| compressed | 16 | 16 | heterogeneous | 2 | 28.22 | 28.05–28.38 | 591.0 |
| compressed | 16 | 16 | segmented | 2 | 28.15 | 28.06–28.25 | 606.4 |
| mixed_high_low | 8 | 8 | heterogeneous | 2 | 22.54 | 22.44–22.64 | 757.4 |
| mixed_high_low | 8 | 8 | segmented | 2 | 21.87 | 21.63–22.11 | 784.9 |
| verified_r64 | 4 | 8 | heterogeneous | 2 | 21.65 | 21.14–22.15 | 789.9 |
| verified_r64 | 4 | 8 | segmented | 2 | 20.91 | 20.90–20.91 | 816.7 |

## Same batch kernel comparisons

- compressed, 4 adapters, batch cap 8: 1.000x throughput (heterogeneous versus segmented).
- compressed, 8 adapters, batch cap 8: 1.240x throughput (heterogeneous versus segmented).
- compressed, 16 adapters, batch cap 8: 1.090x throughput (heterogeneous versus segmented).
- compressed, 16 adapters, batch cap 16: 1.002x throughput (heterogeneous versus segmented).
- mixed_high_low, 8 adapters, batch cap 8: 1.031x throughput (heterogeneous versus segmented).
- verified_r64, 4 adapters, batch cap 8: 1.035x throughput (heterogeneous versus segmented).

## Numerical parity

Maximum same-batch action difference: 0.00000000. This is fixed-input parity, not rollout validation.

## Profiling (separate instrumented runs)

### segmented

| Region | CPU total ms | GPU total ms | Calls |
|---|---:|---:|---:|
| fleet.base | 0.00 | 108.83 | 640 |
| fleet.lora | 0.00 | 48.81 | 621 |

### heterogeneous

| Region | CPU total ms | GPU total ms | Calls |
|---|---:|---:|---:|
| fleet.base | 0.00 | 108.66 | 640 |
| fleet.lora | 0.00 | 42.25 | 621 |

### base_only_probe

| Region | CPU total ms | GPU total ms | Calls |
|---|---:|---:|---:|
| fleet.base | 0.00 | 104.52 | 640 |


## Isolated GPU rank-change compilation probe

After warming rank 1, varying active ranks through [1, 2, 3, 4, 7, 8, 13, 16, 17, 25, 31, 32] created 22 additional old segmented variants and 0 new heterogeneous variants. This proves rank-stride specialization is removed; it does not identify every source of serving latency spikes.

Only device-only region records are displayed above; CPU-associated records of the same name are excluded to avoid double-counting. Profiling totals can include nested operations; they are not added across all profiler events. The base-only probe is a performance ceiling measurement, not a valid policy. Two repetitions provide an initial range, not a confidence interval. HTTP/network overhead is excluded.
