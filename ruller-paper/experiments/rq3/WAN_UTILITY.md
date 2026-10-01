# Wan2.1 downstream utility evaluation

This evaluation replaces the old one-text-probe CLIP pilot. It generates
strictly paired base, full-adapter and energy-truncated videos, then evaluates
the saved videos independently of generation.

The initial single-GPU pool contains PanoWan, UltraWan-1K, Aesthetics,
HighResFix and SpeedControl. UltraWan-4K remains in the protocol but is excluded
from the single-GPU Slurm array because its native 2160x3840 operating point
requires the authors' multi-GPU USP path. Evaluating it at 480x832 would not
measure its advertised downstream function.

## Headroom pilot

```bash
sbatch --array=0-9%1 experiments/rq3/run_wan_utility_pilot.slurm
sbatch experiments/rq3/run_wan_utility_score.slurm
```

The cheap pilot uses one prompt, one seed, 33 frames, and 25 denoising steps at
each adapter's target spatial resolution. The scoring job should be submitted
after all generation jobs finish. An
adapter passes the automatic headroom gate only when its mean full-minus-base
improvement is positive and at least 75% of paired prompt/seed observations
have the expected sign. Inspect effect size and generated clips before running
the compression sweep.

## Compression sweep

Call `wan_utility_generate.py` with `--variant e99`, `e95`, `e90`, `e80`,
`e70`, or `e50` for adapters that pass the gate. Use the same prompt indices and
seeds as base/full. `summarize_wan_utility.py` reports

```
retained = (metric_compressed - metric_base) / (metric_full - metric_base)
```

for every condition.

The bundled OpenCV and CLIP scores are inexpensive screening proxies. Final
paper results should replace `imaging_quality` and `aesthetic_quality` with the
corresponding official VBench dimensions. PanoWan should additionally use its
official cube-map and spherical-continuity evaluator when available. Saved MP4
files and JSON sidecars are sufficient for those evaluators; no generation has
to be repeated.
