#!/usr/bin/env python3
"""Prediction-space local divergence for the video diffusion transformer.

The weight-space perturbation L_W is identical across task families by
construction -- it is sqrt(1 - energy).  What differs is what the model's
execution dynamics do with it.  The first half of that question is how much a
single prediction moves: feed the *same* latent and timestep to the original and
the compressed adapter and compare the transformer's output.

    D_v(t) = || v_c(x_t, t) - v_o(x_t, t) || / || v_o(x_t, t) ||

No sampling loop, no VAE, no text encoder: one forward per adapter per timestep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_inputs(transformer, batch, frames, device, dtype, generator):
    cfg = transformer.config
    hidden = torch.randn(
        batch, frames, cfg.in_channels,
        cfg.sample_height // 1, cfg.sample_width // 1,
        generator=generator, device=device, dtype=dtype)
    encoder = torch.randn(batch, 226, cfg.text_embed_dim,
                          generator=generator, device=device, dtype=dtype)
    return hidden, encoder


def rotary(transformer, frames, device, dtype):
    """CogVideoX-5B uses 3D RoPE; the pipeline builds it, so replicate it here."""
    cfg = transformer.config
    if not getattr(cfg, "use_rotary_positional_embeddings", False):
        return None
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
    p = cfg.patch_size
    grid_h = cfg.sample_height // p
    grid_w = cfg.sample_width // p
    base_h = cfg.sample_height // p
    base_w = cfg.sample_width // p
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=cfg.attention_head_dim,
        crops_coords=((0, 0), (grid_h, grid_w)),
        grid_size=(grid_h, grid_w),
        temporal_size=frames,
    )
    return freqs_cos.to(device=device, dtype=dtype), freqs_sin.to(device=device, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--variant", type=Path, nargs="+", required=True)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[999, 750, 500, 250, 50])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from diffusers import CogVideoXTransformer3DModel
    device, dtype = "cuda", torch.bfloat16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=dtype).to(device).eval()
    print(f"loaded transformer: in_channels={transformer.config.in_channels} "
          f"text_embed_dim={transformer.config.text_embed_dim} "
          f"rope={getattr(transformer.config,'use_rotary_positional_embeddings',False)}", flush=True)

    rope = rotary(transformer, args.frames, device, dtype)

    from peft import PeftModel

    def run(adapter, hidden, encoder, t):
        # These adapters are in PEFT format (adapter_model.safetensors); the
        # diffusers LoRA loader looks for its own file names instead, so load
        # them with the library that wrote them.
        model = PeftModel.from_pretrained(transformer, str(adapter), adapter_name="probe")
        model.eval()
        with torch.no_grad():
            out = model(hidden_states=hidden, encoder_hidden_states=encoder,
                        timestep=t, image_rotary_emb=rope, return_dict=False)[0]
        model.unload()
        del model
        torch.cuda.empty_cache()
        return out.float()

    rows = []
    print(f"\n{'variant':8s} {'timestep':>9s} {'D_v mean':>10s} {'std':>9s}")
    for rep in range(args.repeats):
        gen = torch.Generator(device=device).manual_seed(rep)
        hidden, encoder = build_inputs(transformer, args.batch, args.frames, device, dtype, gen)
        for ts in args.timesteps:
            t = torch.full((args.batch,), ts, device=device, dtype=dtype)
            ref = run(args.original, hidden, encoder, t)
            denom = ref.norm().clamp_min(1e-8)
            for path in args.variant:
                got = run(path, hidden, encoder, t)
                d = float((got - ref).norm() / denom)
                rows.append({"variant": path.name, "timestep": ts, "repeat": rep, "d_v": d})
    for path in args.variant:
        for ts in args.timesteps:
            vals = [r["d_v"] for r in rows if r["variant"] == path.name and r["timestep"] == ts]
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
            print(f"{path.name:8s} {ts:9d} {mu:10.5f} {sd:9.5f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
