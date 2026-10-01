#!/usr/bin/env python3
"""Generate paired Wan utility videos for base/full/truncated adapters.

The output layout is deliberately evaluator-agnostic. Every mp4 has a JSON
sidecar containing the exact prompt, seed, generation settings, adapter and
rank condition, so VBench or a task-specific evaluator can be run later without
regenerating clips.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_task_metrics import b_key, load_lora, to_base, truncate  # noqa: E402


def parse_variant(value: str) -> tuple[str, float | None]:
    if value in {"base", "full"}:
        return value, None
    if value.startswith("e") and value[1:].isdigit():
        tau = int(value[1:]) / 100.0
        if 0.0 < tau <= 1.0:
            return value, tau
    raise argparse.ArgumentTypeError("variant must be base, full, or eXX (for example e90)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--variant", type=parse_variant, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--prompt-indices", type=int, nargs="*")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--height", type=int, help="explicit pilot override")
    ap.add_argument("--width", type=int, help="explicit pilot override")
    ap.add_argument("--frames", type=int, help="explicit pilot override")
    ap.add_argument("--steps", type=int, help="explicit pilot override")
    args = ap.parse_args()

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    protocol = json.loads(args.protocol.read_text())
    pool = json.loads(args.pool.read_text())
    if args.adapter not in protocol["adapters"]:
        raise SystemExit(f"adapter {args.adapter!r} is absent from protocol")
    if args.adapter not in pool:
        raise SystemExit(f"adapter {args.adapter!r} is absent from pool")
    spec = protocol["adapters"][args.adapter]
    variant, tau = args.variant

    vae = AutoencoderKLWan.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.base, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    transformer = pipe.transformer
    state_keys = set(transformer.state_dict())

    weights = None if variant == "base" else load_lora(Path(pool[args.adapter]["path"]))
    pairs = []
    if weights is not None:
        a_keys = sorted(k for k in weights if ".lora_A" in k or ".lora_down" in k)
        pairs = [(a, b_key(a), to_base(a) + ".weight") for a in a_keys]
        pairs = [(a, b, target) for a, b, target in pairs if b in weights and target in state_keys]
        if not pairs:
            raise RuntimeError("no LoRA pairs mapped onto the Wan transformer")
    target_keys = {target for _, _, target in pairs}
    pristine = {k: transformer.state_dict()[k].detach().cpu().clone() for k in target_keys}

    @torch.no_grad()
    def install() -> dict[str, float | int]:
        state = transformer.state_dict()
        kept = total = 0
        kept_energy = dropped_energy = 0.0
        scale = float(spec.get("lora_scale", 1.0))
        for key, tensor in pristine.items():
            state[key].copy_(tensor.to(device=state[key].device, dtype=state[key].dtype))
        if weights is None:
            return {"kept_rank": 0, "total_rank": 0, "rank_fraction": 0.0, "L_W": 0.0}
        for a_key, bkey, target in pairs:
            a, b = weights[a_key].float(), weights[bkey].float()
            total += a.shape[0]
            if tau is None:
                new_a, new_b = a, b
                rank, ek, ed = a.shape[0], float((b @ a).square().sum()), 0.0
            else:
                new_a, new_b, rank, ek, ed = truncate(a, b, tau, None)
            delta = (new_b @ new_a) * scale
            state[target].add_(delta.to(device=state[target].device, dtype=state[target].dtype))
            kept += rank
            kept_energy += ek
            dropped_energy += ed
        denom = kept_energy + dropped_energy
        return {
            "kept_rank": kept,
            "total_rank": total,
            "rank_fraction": kept / total,
            "L_W": math.sqrt(dropped_energy / denom) if denom else 0.0,
            "mapped_modules": len(pairs),
            "lora_scale": scale,
        }

    compression = install()
    height = args.height or int(spec["height"])
    width = args.width or int(spec["width"])
    frames = args.frames or int(spec["num_frames"])
    steps = args.steps or int(spec["steps"])
    indices = args.prompt_indices if args.prompt_indices is not None else range(len(spec["prompts"]))
    out_dir = args.output_root / args.adapter / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    for prompt_index in indices:
        prompt = spec["prompts"][prompt_index]
        for seed in args.seeds:
            stem = f"p{prompt_index:03d}_s{seed:04d}"
            video_path = out_dir / f"{stem}.mp4"
            started = time.perf_counter()
            generated = pipe(
                prompt=prompt,
                negative_prompt=protocol["negative_prompt"],
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=steps,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).frames[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            export_to_video(generated, str(video_path), fps=int(spec["fps"]))
            record = {
                "adapter": args.adapter,
                "repo": pool[args.adapter]["repo"],
                "family": spec["family"],
                "primary_metric": spec["primary_metric"],
                "higher_is_better": spec["higher_is_better"],
                "variant": variant,
                "tau": tau,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "seed": seed,
                "height": height,
                "width": width,
                "num_frames": frames,
                "fps": int(spec["fps"]),
                "steps": steps,
                "generation_seconds": elapsed,
                "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                "compression": compression,
                "video": str(video_path),
            }
            video_path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
            print("RESULT " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
