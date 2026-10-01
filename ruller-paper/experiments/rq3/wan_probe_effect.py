#!/usr/bin/env python3
"""Did the adapter change the video, and can the CLIP probe see it?

The headroom pilot returned differences of -0.010 to +0.003 in CLIP alignment,
which is either "the adapter does nothing" or "the probe cannot see what the
adapter does". Those have opposite consequences and the pilot cannot tell them
apart, because it recorded only the mean over four prompts.

This records both halves for the same prompt and seed: the pixel-level change
between the base and adapted videos, which says whether the adapter acted at all,
and the per-prompt CLIP scores with their spread, which says whether a difference
of 0.01 is inside the noise.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_task_metrics import DOMAIN, to_base, b_key, load_lora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from diffusers import AutoencoderKLWan, WanPipeline
    from transformers import CLIPModel, CLIPProcessor

    vae = AutoencoderKLWan.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.base, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    tf = pipe.transformer
    clip = CLIPModel.from_pretrained(args.clip).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(args.clip)
    print("loaded", flush=True)

    dom = DOMAIN[args.adapter]
    pool = json.loads(args.pool.read_text())
    w = load_lora(Path(pool[args.adapter]["path"]))
    orig = {k: v.detach().clone() for k, v in tf.state_dict().items()}

    def score(frames, probe):
        imgs = [np.asarray(f) for f in frames[:: max(len(frames) // 8, 1)]]
        inp = proc(text=[probe], images=imgs, return_tensors="pt", padding=True,
                   truncation=True).to("cuda")
        with torch.no_grad():
            o = clip(**inp)
        im = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
        tx = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
        return float((im @ tx.T).mean())

    def gen(prompt, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        out = pipe(prompt=prompt, height=480, width=832, num_frames=args.frames,
                   num_inference_steps=args.steps, generator=g)
        return out.frames[0]

    def apply_full():
        sd = tf.state_dict()
        n = 0
        for a in sorted(k for k in w if ".lora_A" in k or ".lora_down" in k):
            b = b_key(a)
            key = to_base(a) + ".weight"
            if b not in w or key not in orig:
                continue
            delta = (w[b].float() @ w[a].float()).to(device=orig[key].device,
                                                    dtype=orig[key].dtype)
            sd[key].copy_(orig[key] + delta)
            n += 1
        return n

    def restore():
        sd = tf.state_dict()
        for k, v in orig.items():
            sd[k].copy_(v)

    rows = []
    for i, p in enumerate(dom["prompts"][: args.prompts]):
        restore()
        fb = gen(p, i)
        n = apply_full()
        ff = gen(p, i)
        A = np.stack([np.asarray(x, dtype=np.float32) for x in fb])
        B = np.stack([np.asarray(x, dtype=np.float32) for x in ff])
        rel = float(np.linalg.norm(B - A) / max(np.linalg.norm(A), 1e-8))
        sb, sf = score(fb, dom["probe"]), score(ff, dom["probe"])
        rows.append(dict(prompt=p, modules=n, pixel_rel_change=rel,
                         clip_base=sb, clip_full=sf, delta=sf - sb))
        print(f"[{i}] modules={n} pixel_rel_change={rel:.4f}  "
              f"clip base={sb:.4f} full={sf:.4f} delta={sf-sb:+.4f}", flush=True)
    restore()
    d = np.array([r["delta"] for r in rows])
    px = np.array([r["pixel_rel_change"] for r in rows])
    print(f"\npixel change: mean {px.mean():.4f}  (0 would mean the adapter never applied)")
    print(f"CLIP delta:   mean {d.mean():+.4f}  sd {d.std(ddof=1) if len(d)>1 else float('nan'):.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
