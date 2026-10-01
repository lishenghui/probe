#!/usr/bin/env python3
"""Generate the matched, unmodified Wan 2.1 T2V 1.3B baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, paintings, worst quality, "
    "low quality, JPEG artifacts, ugly, incomplete, deformed, disfigured, misshapen limbs, "
    "still picture, messy background, walking backwards"
)
DEFAULT_PROMPT = (
    "Cinematic wildlife footage of a red fox sprinting through a snowy pine forest at dawn. "
    "The camera tracks smoothly alongside the fox at low angle; powder snow sprays from its paws, "
    "its fur moves naturally in the wind, and warm sunrise rays stream between the trees. "
    "Realistic motion, detailed fur, shallow depth of field, dynamic continuous shot."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        args.model, vae=vae, torch_dtype=torch.bfloat16
    ).to("cuda")

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    frames = pipe(
        prompt=args.prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=args.steps,
        guidance_scale=5.0,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).frames[0]
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    export_to_video(frames, str(args.output), fps=16)

    result = {
        "model": args.model,
        "adapter": None,
        "prompt": args.prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": args.seed,
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "num_inference_steps": args.steps,
        "guidance_scale": 5.0,
        "fps": 16,
        "inference_seconds": inference_seconds,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
