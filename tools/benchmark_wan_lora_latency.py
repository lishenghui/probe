#!/usr/bin/env python3
"""End-to-end denoising latency of Wan2.1-1.3B against FraQ-compressed sidecars.

The Wan AnyFlow adapter is a rank-256 LoRA on every linear layer, so the sidecar
is a large fraction of a denoising step -- which makes it the case where rank
compression should show up end to end.  Historically it did not below e90: the
curve flattened at ~7.5 s no matter how far the rank fell.

For each energy target this times the pipeline's denoising loop (VAE decode is
skipped via ``output_type="latent"``) under two sidecar implementations:

  peft       -- stock PEFT, three passes over the activation per layer
  loraforge  -- enable_loraforge_peft, sidecar folded into the base GEMM

``none`` disables the adapter entirely and is the floor a rank-proportional
sidecar should approach as the retained rank drops.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

PROMPT = (
    "Cinematic wildlife footage of a red fox sprinting through a snowy pine forest at dawn. "
    "The camera tracks smoothly alongside the fox at low angle; powder snow sprays from its paws."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--scheduler-model", default="nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--stem", default="anyflow-wan2.1-t2v-1.3b_all-linear_rank256_anyflow-sidecar")
    parser.add_argument("--original-adapter", required=True)
    parser.add_argument("--energies", nargs="+", default=["e95", "e90", "e80", "e50"])
    parser.add_argument("--aligned-dir", type=Path, default=None,
                        help="second set of the same energies whose per-module ranks were "
                             "rounded up to a GEMM-friendly multiple; loaded alongside the "
                             "first so both are timed in one process at one clock")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    return parser.parse_args()


def time_generation(pipe, args, warmup, reps):
    samples = []
    for index in range(warmup + reps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        pipe(
            prompt=PROMPT,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            output_type="latent",
            generator=torch.Generator("cuda").manual_seed(0),
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index >= warmup:
            samples.append(elapsed)
    return statistics.median(samples), min(samples)


def main() -> None:
    args = parse_args()
    from far.models.transformer_far_wan_model import FAR_Wan_Transformer3DModel
    from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline
    from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler
    from monkeypatch import load_anyflow_lora

    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    transformer = FAR_Wan_Transformer3DModel.from_pretrained(
        args.base_model, subfolder="transformer", torch_dtype=dtype
    )

    adapters = [("e100", args.original_adapter)]
    for energy in args.energies:
        path = args.adapter_dir / f"{args.stem}_fraq_{energy}.safetensors"
        if path.is_file():
            adapters.append((energy, str(path)))
        else:
            print(f"skipping {energy}: {path} not found", flush=True)
    if args.aligned_dir:
        for energy in args.energies:
            path = args.aligned_dir / f"{args.stem}_fraq_{energy}.safetensors"
            if path.is_file():
                adapters.append((f"{energy}A", str(path)))
            else:
                print(f"skipping {energy}A: {path} not found", flush=True)
    for name, path in adapters:
        load_anyflow_lora(transformer, path, adapter_name=name)
        print(f"loaded {name}", flush=True)

    ranks = {}
    for name, _ in adapters:
        values = [module.lora_A[name].weight.shape[0]
                  for module in transformer.modules()
                  if name in getattr(module, "lora_A", {})]
        ranks[name] = {"modules": len(values), "rank_mean": sum(values) / max(len(values), 1),
                       "rank_min": min(values, default=0), "rank_max": max(values, default=0)}
        print(f"{name}: {ranks[name]}", flush=True)

    scheduler = FlowMapDiscreteScheduler.from_pretrained(args.scheduler_model, subfolder="scheduler")
    pipe = WanAnyFlowPipeline.from_pretrained(
        args.base_model, transformer=transformer, scheduler=scheduler, torch_dtype=dtype
    ).to("cuda", dtype=dtype)

    results = []
    for mode in ("peft", "loraforge"):
        if mode == "loraforge":
            from loraforge_kernels import enable_loraforge_peft
            print(f"\npatched {enable_loraforge_peft(pipe.transformer)} LoRA linears", flush=True)
        print(f"\n=== sidecar: {mode} ===", flush=True)

        pipe.transformer.disable_adapters()
        floor, floor_min = time_generation(pipe, args, args.warmup, args.reps)
        pipe.transformer.enable_adapters()
        print(f"{'variant':8s} {'rank':>7s} {'median s':>9s} {'min s':>8s} {'vs none':>9s} {'vs e100':>8s}")
        print(f"{'none':8s} {0:7.1f} {floor:9.3f} {floor_min:8.3f}")
        results.append({"mode": mode, "variant": "none", "rank_mean": 0.0,
                        "median_seconds": floor, "min_seconds": floor_min})

        reference = None
        for name, _ in adapters:
            pipe.transformer.set_adapter(name)
            median, fastest = time_generation(pipe, args, args.warmup, args.reps)
            reference = reference or median
            print(f"{name:8s} {ranks[name]['rank_mean']:7.1f} {median:9.3f} {fastest:8.3f} "
                  f"{100 * (median - floor) / floor:8.1f}% {reference / median:7.2f}x", flush=True)
            results.append({
                "mode": mode, "variant": name, "rank_mean": ranks[name]["rank_mean"],
                "median_seconds": median, "min_seconds": fastest,
                "sidecar_pct_of_none": 100 * (median - floor) / floor,
                "speedup_vs_e100": reference / median,
            })
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(
                {"gpu": torch.cuda.get_device_name(0), "steps": args.steps,
                 "ranks": ranks, "rows": results}, indent=2) + "\n")

    print("\n=== loraforge vs peft ===")
    by_key = {(r["mode"], r["variant"]): r["median_seconds"] for r in results}
    for name in ["none"] + [n for n, _ in adapters]:
        peft_s, forge_s = by_key.get(("peft", name)), by_key.get(("loraforge", name))
        if peft_s and forge_s:
            print(f"{name:8s} {peft_s:6.3f}s -> {forge_s:6.3f}s   {100 * (peft_s - forge_s) / peft_s:5.1f}% faster")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
