#!/usr/bin/env python3
"""Summarise the hook-vs-fused MotionLoRA A/B into a rank-scaling table.

The question this answers: does end-to-end latency actually fall as FraQ drops
the retained rank?  For each sidecar implementation it reports latency per
variant, the speedup over the uncompressed E100 adapter, and how much of the
sidecar's cost over the un-adapted BASE run survives.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ORDER = ["BASE", "E100", "E95", "E95A", "E90", "E90A", "E80", "E80A", "E70", "E70A"]


def load(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    grouped: dict[str, list[float]] = {}
    for record in json.loads(path.read_text()):
        # The first sample of each variant pays for CUDA graph capture, Triton
        # autotuning and allocator growth, so it is warm-up, not signal.
        grouped.setdefault(record["variant"], []).append(
            record.get("denoising_seconds", record["inference_seconds"])
        )
    return {name: statistics.median(values[1:] or values) for name, values in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = {label: load(args.run_dir / f"raw_timings_{label}.json") for label in ("hook", "fused")}
    summary = {"run_dir": str(args.run_dir), "modes": {}}

    for label, timings in runs.items():
        if not timings:
            continue
        base = timings.get("BASE")
        full = timings.get("E100")
        print(f"\n=== sidecar: {label} ===")
        header = f"{'variant':8s} {'seconds':>9s} {'vs E100':>9s} {'sidecar cost':>13s}"
        print(header)
        rows = {}
        for name in ORDER:
            if name not in timings:
                continue
            seconds = timings[name]
            speedup = full / seconds if full else float("nan")
            overhead = (100.0 * (seconds - base) / base) if base else float("nan")
            rows[name] = {"seconds": seconds, "speedup_vs_e100": speedup, "sidecar_pct_of_base": overhead}
            print(f"{name:8s} {seconds:9.2f} {speedup:8.2f}x {overhead:12.1f}%")
        summary["modes"][label] = rows

    if runs.get("hook") and runs.get("fused"):
        print("\n=== fused vs hook, per variant ===")
        for name in ORDER:
            hook, fused = runs["hook"].get(name), runs["fused"].get(name)
            if hook and fused:
                print(f"{name:8s} {hook:7.2f}s -> {fused:7.2f}s   {100 * (hook - fused) / hook:5.1f}% faster")

    out = args.run_dir / "fused_lora_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
