# OFT context execution experiment

Backend: HF context-only execution with CUDA Graph, NOT vLLM or TensorRT.

Maximum fixed-input action difference: 0.00000000.
This does not establish rollout quality. No model weights were compressed.

| Mode | Repeats | Median req/s | Range req/s | Median p95 ms |
|---|---:|---:|---:|---:|
| reference | 2 | 23.56 | 23.25–23.88 | 740.3 |
| context | 2 | 24.19 | 23.67–24.72 | 719.6 |
| graph | 2 | 24.76 | 24.65–24.87 | 695.5 |

Serving uses 16 concurrent clients, four r64 adapters, batch cap 8,
and adapter switching on every request. HTTP/network overhead is excluded.
Two repeats provide an initial range, not a confidence interval.
Capture counts include client warmup; see the JSON for individual timings.
Total captures: 2; replays: 52.
