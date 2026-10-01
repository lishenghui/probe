# Paired dynamic serving optimization

GPU: NVIDIA GH200 120GB. Fixed observation, 16 concurrent clients, same banks across modes. Legacy compressed banks are performance-only.

Selected split-K: 1; calibration: [{'split_k': 1, 'median_seconds': 0.30890939105302095}, {'split_k': 4, 'median_seconds': 0.33240780187770724}]

| Family | Adapters | Max batch | Mode | Runs | Median req/s | Range req/s | Median p95 ms |
|---|---:|---:|---|---:|---:|---|---:|
| compressed | 4 | 8 | grouped | 2 | 15.99 | 15.15–16.82 | 1090.0 |
| compressed | 4 | 8 | ragged | 2 | 14.31 | 14.13–14.50 | 1208.2 |
| compressed | 4 | 8 | segmented | 2 | 22.72 | 22.27–23.17 | 763.7 |
| compressed | 8 | 8 | grouped | 2 | 8.94 | 8.83–9.06 | 1853.0 |
| compressed | 8 | 8 | ragged | 2 | 14.23 | 14.15–14.30 | 1184.9 |
| compressed | 8 | 8 | segmented | 2 | 22.32 | 22.22–22.41 | 765.0 |
| compressed | 16 | 8 | base_only_probe | 1 | 31.60 | 31.60–31.60 | 545.9 |
| compressed | 16 | 8 | grouped | 2 | 4.22 | 4.17–4.26 | 3902.3 |
| compressed | 16 | 8 | ragged | 2 | 14.07 | 14.07–14.08 | 1211.8 |
| compressed | 16 | 8 | segmented | 2 | 16.11 | 14.41–17.82 | 2439.4 |
| compressed | 16 | 16 | ragged | 2 | 16.06 | 15.88–16.23 | 1010.9 |
| compressed | 16 | 16 | segmented | 2 | 28.76 | 28.67–28.86 | 568.9 |
| mixed_high_low | 8 | 8 | grouped | 2 | 8.44 | 8.29–8.58 | 1971.1 |
| mixed_high_low | 8 | 8 | ragged | 2 | 13.29 | 13.20–13.39 | 1236.5 |
| mixed_high_low | 8 | 8 | segmented | 2 | 22.01 | 21.99–22.03 | 763.7 |
| verified_r64 | 4 | 8 | grouped | 2 | 15.29 | 14.90–15.68 | 1124.4 |
| verified_r64 | 4 | 8 | ragged | 2 | 12.87 | 12.76–12.98 | 1312.8 |
| verified_r64 | 4 | 8 | segmented | 2 | 21.76 | 21.44–22.07 | 825.9 |

## Same batch kernel comparisons

- compressed, 4 adapters, batch cap 8: 1.587x throughput versus token-wise ragged.
- compressed, 8 adapters, batch cap 8: 1.569x throughput versus token-wise ragged.
- compressed, 16 adapters, batch cap 8: 1.145x throughput versus token-wise ragged.
- compressed, 16 adapters, batch cap 16: 1.792x throughput versus token-wise ragged.
- mixed_high_low, 8 adapters, batch cap 8: 1.655x throughput versus token-wise ragged.
- verified_r64, 4 adapters, batch cap 8: 1.691x throughput versus token-wise ragged.

## Numerical parity

Maximum same-batch action difference: 0.02057321. This is fixed-input parity, not rollout validation.

## Profiling (separate instrumented runs)

### ragged

| Region | CPU total ms | GPU total ms | Calls |
|---|---:|---:|---:|
| fleet.lora | 0.00 | 334.98 | 621 |
| fleet.base | 95.27 | 111.19 | 640 |
| fleet.base | 0.00 | 95.61 | 640 |
| fleet.lora | 205.55 | 3.40 | 640 |

### segmented

| Region | CPU total ms | GPU total ms | Calls |
|---|---:|---:|---:|
| fleet.base | 0.00 | 104.86 | 640 |
| fleet.base | 75.50 | 103.67 | 640 |
| fleet.lora | 0.00 | 47.06 | 621 |

### base_only_probe

| Region | CPU total ms | GPU total ms | Calls |
|---|---:|---:|---:|
| fleet.base | 0.00 | 106.22 | 640 |
| fleet.base | 74.97 | 104.90 | 640 |

Profiling totals can include nested operations; they are not added across all profiler events. The base-only probe is a performance ceiling measurement, not a valid policy. Two repetitions provide an initial range, not a confidence interval. HTTP/network overhead is excluded.
