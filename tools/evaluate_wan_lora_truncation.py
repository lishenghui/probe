#!/usr/bin/env python3
"""Generate matched Wan2.2 videos while dynamically switching LoRA ranks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from lightx2v import LightX2VPipeline


PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_high = args.adapter_root / "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors"
    original_low = args.adapter_root / "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors"
    truncated = args.adapter_root / "fraq_truncated"
    variants = [("rank64", original_high, original_low)]
    for rank in (48, 32, 16):
        variants.append(
            (
                f"rank{rank}",
                truncated / f"{original_high.stem}_fraq_rank{rank}.safetensors",
                truncated / f"{original_low.stem}_fraq_rank{rank}.safetensors",
            )
        )
    for _, high, low in variants:
        if not high.is_file() or not low.is_file():
            raise FileNotFoundError(f"Missing adapter pair: {high}, {low}")

    pipe = LightX2VPipeline(model_path=str(args.model_path), model_cls="wan2.2_moe_distill", task="t2v")
    pipe.enable_lora(
        [
            {"name": "high_noise_model", "path": str(original_high), "strength": 1.0},
            {"name": "low_noise_model", "path": str(original_low), "strength": 1.0},
        ],
        lora_dynamic_apply=True,
    )
    pipe.enable_offload(cpu_offload=True, offload_granularity="model")
    # Apply once explicitly as well as through create_generator.  This keeps
    # required scheduler fields visible across LightX2V schema revisions.
    infer_config = json.loads(args.config.read_text())
    pipe.update(infer_config)
    # LightX2V's set_args2config filters input-info fields such as
    # target_video_length.  Keeping config_json on the pipeline lets
    # auto_calc_config merge those fields back, as its CLI path does.
    pipe.config_json = str(args.config)
    print("INFER_CONFIG " + json.dumps(infer_config, sort_keys=True), flush=True)
    print(f"PIPE_TARGET_VIDEO_LENGTH {pipe.target_video_length}", flush=True)
    pipe.create_generator(config_json=str(args.config))

    measurements = []
    for index, (name, high, low) in enumerate(variants):
        switch_seconds = 0.0
        if index:
            switch_start = time.perf_counter()
            pipe.runner.switch_lora(high_lora_path=str(high), low_lora_path=str(low))
            torch.cuda.synchronize()
            switch_seconds = time.perf_counter() - switch_start
        output = args.output_dir / f"wan22_t2v_{name}_seed42.mp4"
        torch.cuda.synchronize()
        started = time.perf_counter()
        pipe.generate(
            seed=42,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            save_result_path=str(output),
        )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        result = {
            "variant": name,
            "high_lora": str(high),
            "low_lora": str(low),
            "seed": 42,
            "prompt": PROMPT,
            "switch_seconds": switch_seconds,
            "inference_and_encode_seconds": inference_seconds,
            "output": str(output),
        }
        measurements.append(result)
        print("MEASUREMENT " + json.dumps(result, ensure_ascii=False), flush=True)
        (args.output_dir / "timings.json").write_text(json.dumps(measurements, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
