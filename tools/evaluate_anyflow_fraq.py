#!/usr/bin/env python3
"""Generate matched samples for the full and FRAQ-compressed AnyFlow LoRAs."""

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


PROMPT = (
    "Cinematic wildlife footage of a red fox sprinting through a snowy pine forest at dawn. "
    "The camera tracks smoothly alongside the fox at low angle; powder snow sprays from its paws, "
    "its fur moves naturally in the wind, and warm sunrise rays stream between the trees. "
    "Realistic motion, detailed fur, shallow depth of field, dynamic continuous shot."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--scheduler-model", required=True)
    parser.add_argument("--original-adapter", required=True)
    parser.add_argument("--compressed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = "anyflow-wan2.1-t2v-1.3b_all-linear_rank256_anyflow-sidecar"
    variants = [
        ("full", args.original_adapter),
        ("e90", str(args.compressed_dir / f"{stem}_fraq_e90.safetensors")),
        ("e80", str(args.compressed_dir / f"{stem}_fraq_e80.safetensors")),
        ("e50", str(args.compressed_dir / f"{stem}_fraq_e50.safetensors")),
    ]
    for name, path in variants[1:]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name} adapter: {path}")

    transformer = FAR_Wan_Transformer3DModel.from_pretrained(
        args.base_model, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    loaded = []
    for name, path in variants:
        loaded_path = load_anyflow_lora(transformer, path, adapter_name=name)
        loaded.append((name, str(loaded_path)))
    scheduler = FlowMapDiscreteScheduler.from_pretrained(args.scheduler_model, subfolder="scheduler")
    pipe = WanAnyFlowPipeline.from_pretrained(
        args.base_model,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to("cuda", dtype=torch.bfloat16)

    measurements = []
    for name, path in loaded:
        pipe.transformer.set_adapter(name)
        torch.cuda.reset_peak_memory_stats()
        output = args.output_dir / f"anyflow_fraq_{name}_fox_seed0.mp4"
        started = time.perf_counter()
        frames = pipe(
            prompt=PROMPT,
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=4,
            generator=torch.Generator("cuda").manual_seed(0),
        ).frames[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        export_to_video(frames, str(output), fps=16)
        result = {
            "variant": name,
            "adapter": path,
            "prompt": PROMPT,
            "seed": 0,
            "num_inference_steps": 4,
            "inference_seconds": seconds,
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
            "output": str(output),
        }
        measurements.append(result)
        print("RESULT " + json.dumps(result), flush=True)
        (args.output_dir / "fraq_timings.json").write_text(json.dumps(measurements, indent=2) + "\n")


if __name__ == "__main__":
    main()
