#!/usr/bin/env python3
"""Where does the LoRA sidecar's time actually go on Wan2.1?

Back-of-envelope models of the sidecar cost have repeatedly disagreed with the
end-to-end measurement, so this stops estimating and reads the CUDA kernel
timeline instead.  It profiles one denoising step per adapter and reports CUDA
time aggregated by kernel, so the sidecar's share and its scaling with rank are
visible directly rather than inferred.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

PROMPT = "Cinematic wildlife footage of a red fox sprinting through a snowy pine forest at dawn."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--scheduler-model", default="nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, default=None)
    parser.add_argument("--stem", default="anyflow-wan2.1-t2v-1.3b_all-linear_rank256_anyflow-sidecar")
    parser.add_argument("--original-adapter", required=True)
    parser.add_argument("--variants", nargs="+", default=["e100", "e50A"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def kernel_stats(prof):
    """CUDA microseconds and launch counts per *device* kernel.

    Only rows with no CPU time are counted: the `aten::` wrappers carry the same
    device time as the kernels they launch, so including them double counts.
    """
    times, counts = collections.Counter(), collections.Counter()
    for event in prof.key_averages():
        if event.self_device_time_total and not event.self_cpu_time_total:
            times[event.key] += event.self_device_time_total
            counts[event.key] += event.count
    return times, counts


def run_one(pipe, args, label, results):
    # Repeat: a single profile of this pipeline swings by ~100 ms, mostly in the
    # launch-queue stall, which is larger than the sidecar being measured.
    samples = []
    for _ in range(args.repeats):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            pipe(prompt=PROMPT, height=480, width=832, num_frames=81,
                 num_inference_steps=args.steps, output_type="latent",
                 generator=torch.Generator("cuda").manual_seed(0))
            torch.cuda.synchronize()
        samples.append(kernel_stats(prof))
    # Median per kernel across repeats.
    names = set().union(*(t.keys() for t, _ in samples))
    times = {n: statistics.median([t.get(n, 0) for t, _ in samples]) for n in names}
    counts = {n: statistics.median([c.get(n, 0) for _, c in samples]) for n in names}
    total_us = sum(times.values())
    print(f"\n=== {label}: {total_us/1000:.1f} ms device time, "
          f"{sum(counts.values()):.0f} launches (median of {args.repeats}) ===")
    print(f"{'us':>10s} {'share':>7s} {'launches':>9s}  kernel")
    rows = []
    for name in sorted(times, key=lambda n: -times[n])[:16]:
        print(f"{times[name]:10.0f} {100*times[name]/total_us:6.1f}% "
              f"{counts[name]:9.0f}  {name[:80]}")
        rows.append({"kernel": name, "us": times[name], "launches": counts[name],
                     "share_pct": 100 * times[name] / total_us})
    results[label] = {"total_us": total_us, "total_launches": sum(counts.values()),
                      "top": rows, "all": times, "counts": counts}


def main() -> None:
    args = parse_args()
    from far.models.transformer_far_wan_model import FAR_Wan_Transformer3DModel
    from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline
    from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler
    from monkeypatch import load_anyflow_lora
    from loraforge_kernels import enable_loraforge_peft

    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    transformer = FAR_Wan_Transformer3DModel.from_pretrained(
        args.base_model, subfolder="transformer", torch_dtype=dtype)

    wanted = set(args.variants)
    loaded = []
    if "e100" in wanted:
        load_anyflow_lora(transformer, args.original_adapter, adapter_name="e100")
        loaded.append("e100")
    for name in args.variants:
        if name == "e100":
            continue
        base = name[:-1] if name.endswith("A") else name
        root = args.aligned_dir if name.endswith("A") else args.adapter_dir
        path = root / f"{args.stem}_fraq_{base}.safetensors"
        if path.is_file():
            load_anyflow_lora(transformer, str(path), adapter_name=name)
            loaded.append(name)
        else:
            print(f"skipping {name}: {path} not found", flush=True)

    scheduler = FlowMapDiscreteScheduler.from_pretrained(args.scheduler_model, subfolder="scheduler")
    pipe = WanAnyFlowPipeline.from_pretrained(
        args.base_model, transformer=transformer, scheduler=scheduler, torch_dtype=dtype
    ).to("cuda", dtype=dtype)
    print(f"patched {enable_loraforge_peft(pipe.transformer)} LoRA linears", flush=True)

    results = {}
    # Warm up every path that will be profiled, so compilation and the variant
    # probe do not land inside a profile.
    for name in loaded:
        pipe.transformer.set_adapter(name)
        pipe(prompt=PROMPT, height=480, width=832, num_frames=81,
             num_inference_steps=args.steps, output_type="latent",
             generator=torch.Generator("cuda").manual_seed(0))
    pipe.transformer.disable_adapters()
    pipe(prompt=PROMPT, height=480, width=832, num_frames=81, num_inference_steps=args.steps,
         output_type="latent", generator=torch.Generator("cuda").manual_seed(0))
    torch.cuda.synchronize()

    run_one(pipe, args, "none", results)
    pipe.transformer.enable_adapters()
    for name in loaded:
        pipe.transformer.set_adapter(name)
        run_one(pipe, args, name, results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    print("\n=== sidecar attribution (device us above the no-adapter run) ===")
    floor = results["none"]["all"]
    floor_counts = results["none"]["counts"]
    for name in loaded:
        cur, cur_counts = results[name]["all"], results[name]["counts"]
        delta = {k: cur.get(k, 0) - floor.get(k, 0) for k in set(cur) | set(floor)}
        total = results[name]["total_us"] - results["none"]["total_us"]
        dlaunch = results[name]["total_launches"] - results["none"]["total_launches"]
        print(f"\n{name}: {total/1000:+.1f} ms device time, {dlaunch:+.0f} launches over none")
        for k in sorted(delta, key=lambda n: -abs(delta[n]))[:10]:
            if abs(delta[k]) < 500:
                continue
            dc = cur_counts.get(k, 0) - floor_counts.get(k, 0)
            print(f"  {delta[k]:+9.0f} us {dc:+8.0f} launches  {k[:70]}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
