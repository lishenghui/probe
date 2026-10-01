#!/usr/bin/env python3
"""Generate a Wan 2.1 T2V sample with the extracted AnyFlow LoRA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from far.models.transformer_far_wan_model import FAR_Wan_Transformer3DModel
from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline
from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler
from monkeypatch import load_anyflow_lora


DEFAULT_PROMPT = (
    "Cinematic wildlife footage of a red fox sprinting through a snowy pine forest at dawn. "
    "The camera tracks smoothly alongside the fox at low angle; powder snow sprays from its paws, "
    "its fur moves naturally in the wind, and warm sunrise rays stream between the trees. "
    "Realistic motion, detailed fur, shallow depth of field, dynamic continuous shot."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--scheduler-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    transformer = FAR_Wan_Transformer3DModel.from_pretrained(
        args.base_model, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    adapter_path = load_anyflow_lora(transformer, args.adapter)
    scheduler = FlowMapDiscreteScheduler.from_pretrained(args.scheduler_model, subfolder="scheduler")
    pipe = WanAnyFlowPipeline.from_pretrained(
        args.base_model,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to("cuda", dtype=torch.bfloat16)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    frames = pipe(
        prompt=args.prompt,
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=4,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).frames[0]
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    export_to_video(frames, str(args.output), fps=16)

    result = {
        "base_model": args.base_model,
        "scheduler_model": args.scheduler_model,
        "adapter": str(adapter_path),
        "prompt": args.prompt,
        "seed": args.seed,
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "num_inference_steps": 4,
        "fps": 16,
        "inference_seconds": inference_seconds,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
