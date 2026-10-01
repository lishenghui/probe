#!/usr/bin/env python3
"""Downstream utility for the Wan2.1-T2V-1.3B video pool under truncation.

The other pools in this paper carry gold answers. These four do not: they are
style/domain LoRAs, so "did the adapter do its job" has to be operationalised.
We use CLIP alignment between generated frames and a text probe describing the
adapter's target domain, and we score the un-adapted base model on the same
prompts, so retained utility is the same quantity as everywhere else,

    u = (m_comp - m_base) / (m_full - m_base).

The headroom m_full - m_base is the thing that decides whether this pool is
usable at all. If the proxy metric cannot tell the full adapter from the base
model, u has no denominator and the pool says nothing -- the same filter that
dropped 7 of 48 LoraRetriever adapters. That is why --pilot exists: it runs base
and full only, and is meant to be read before any truncation sweep is launched.

Two key conventions appear in this pool and both map onto the diffusers base:

  native Wan    blocks.0.cross_attn.k.lora_A.default.weight -> blocks.0.attn2.to_k
  PEFT          base_model.model.blocks.0.attn1.to_k.lora_A.weight -> blocks.0.attn1.to_k
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

NATIVE = [
    (r"^blocks\.(\d+)\.self_attn\.q$",  r"blocks.\1.attn1.to_q"),
    (r"^blocks\.(\d+)\.self_attn\.k$",  r"blocks.\1.attn1.to_k"),
    (r"^blocks\.(\d+)\.self_attn\.v$",  r"blocks.\1.attn1.to_v"),
    (r"^blocks\.(\d+)\.self_attn\.o$",  r"blocks.\1.attn1.to_out.0"),
    (r"^blocks\.(\d+)\.cross_attn\.q$", r"blocks.\1.attn2.to_q"),
    (r"^blocks\.(\d+)\.cross_attn\.k$", r"blocks.\1.attn2.to_k"),
    (r"^blocks\.(\d+)\.cross_attn\.v$", r"blocks.\1.attn2.to_v"),
    (r"^blocks\.(\d+)\.cross_attn\.o$", r"blocks.\1.attn2.to_out.0"),
    (r"^blocks\.(\d+)\.ffn\.0$",        r"blocks.\1.ffn.net.0.proj"),
    (r"^blocks\.(\d+)\.ffn\.2$",        r"blocks.\1.ffn.net.2"),
]

# each adapter's target domain, as a CLIP text probe, plus the prompts it is
# exercised on. The probe describes what the adapter was trained to add, so the
# base model should score lower on it than the adapted model does.
DOMAIN = {
    "panowan": dict(
        probe="a 360 degree panoramic equirectangular view",
        prompts=["a panoramic view of a mountain valley at sunrise",
                 "a 360 degree view of a busy city square",
                 "a panoramic shot of waves breaking on a rocky shore",
                 "a wide panoramic view of a forest clearing in autumn"]),
    "ultrawan1k": dict(
        probe="an extremely sharp, highly detailed, high resolution photograph",
        prompts=["a close-up of a hummingbird hovering by a flower",
                 "a detailed shot of rain falling on a window",
                 "a macro view of frost forming on a leaf",
                 "a detailed portrait of a lion in tall grass"]),
    "ultrawan4k": dict(
        probe="an extremely sharp, highly detailed, ultra high resolution photograph",
        prompts=["a close-up of a hummingbird hovering by a flower",
                 "a detailed shot of rain falling on a window",
                 "a macro view of frost forming on a leaf",
                 "a detailed portrait of a lion in tall grass"]),
    "longcat": dict(
        probe="a smooth, temporally coherent, cleanly rendered video",
        prompts=["a cat walking slowly across a wooden floor",
                 "a paper boat drifting down a stream",
                 "a candle flame flickering in a dark room",
                 "a person waving from a train window"]),
}


def to_base(a_key: str) -> str:
    s = a_key
    for pre in ("base_model.model.", "base_model.", "diffusion_model.", "transformer."):
        if s.startswith(pre):
            s = s[len(pre):]
    s = re.sub(r"\.lora_(A|B|down|up)(\.default)?\.weight$", "", s)
    for pat, rep in NATIVE:
        if re.match(pat, s):
            return re.sub(pat, rep, s)
    return s


def b_key(a_key: str) -> str:
    return (a_key.replace(".lora_A", ".lora_B") if ".lora_A" in a_key
            else a_key.replace(".lora_down", ".lora_up"))


def load_lora(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path)
    w = torch.load(path, map_location="cpu", weights_only=False)
    return w.get("state_dict", w) if isinstance(w, dict) else w


def truncate(a: torch.Tensor, b: torch.Tensor, tau: float | None, k_fixed: int | None):
    """Energy- or rank-truncated (A, B) plus the kept rank and discarded energy."""
    a32, b32 = a.float(), b.float()
    qb, rb = torch.linalg.qr(b32, mode="reduced")
    qa, ra = torch.linalg.qr(a32.T, mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    e = s.square()
    if k_fixed is not None:
        k = min(k_fixed, e.numel())
    else:
        c = torch.cumsum(e, 0)
        k = min(int(torch.searchsorted(c, tau * c[-1]).item()) + 1, e.numel())
    nb = qb @ (u[:, :k] * s[:k])
    na = (vh[:k] @ qa.T)
    return na, nb, k, float(e[:k].sum()), float(e[k:].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[0.99, 0.95, 0.90, 0.80, 0.70, 0.50])
    ap.add_argument("--pilot", action="store_true",
                    help="base and full adapter only: the headroom check that decides "
                         "whether a truncation sweep on this pool means anything")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import numpy as np
    from diffusers import AutoencoderKLWan, WanPipeline
    from transformers import CLIPModel, CLIPProcessor

    vae = AutoencoderKLWan.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.base, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    tf = pipe.transformer
    print("pipeline loaded", flush=True)

    clip = CLIPModel.from_pretrained(args.clip).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(args.clip)

    def score(frames, probe: str) -> float:
        """Mean CLIP cosine similarity between the probe text and the frames."""
        imgs = [np.asarray(f) for f in frames[:: max(len(frames) // 8, 1)]]
        inp = proc(text=[probe], images=imgs, return_tensors="pt", padding=True,
                   truncation=True).to("cuda")
        with torch.no_grad():
            out = clip(**inp)
        im = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        tx = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((im @ tx.T).mean())

    def generate(prompts, probe):
        vals = []
        for i, p in enumerate(prompts):
            g = torch.Generator(device="cuda").manual_seed(args.seed + i)
            out = pipe(prompt=p, height=args.height, width=args.width,
                       num_frames=args.frames, num_inference_steps=args.steps,
                       generator=g)
            vals.append(score(out.frames[0], probe))
        return float(np.mean(vals))

    pool = json.loads(args.pool.read_text())
    names = args.adapters or sorted(pool)
    orig = {k: v.detach().clone() for k, v in tf.state_dict().items()}
    results = []

    for name in names:
        dom = DOMAIN[name]
        w = load_lora(Path(pool[name]["path"]))
        a_keys = sorted(k for k in w if ".lora_A" in k or ".lora_down" in k)
        pairs = [(k, b_key(k), to_base(k)) for k in a_keys]
        pairs = [(a, b, t) for a, b, t in pairs if b in w]

        def apply(rule_tau, rule_k):
            sd = tf.state_dict()
            keep = drop = 0.0
            ktot = ntot = 0
            for a, b, tgt in pairs:
                key = tgt + ".weight"
                if key not in orig:
                    continue
                if rule_tau is None and rule_k is None:
                    na, nb = w[a].float(), w[b].float()
                    k, ek, ed = na.shape[0], 0.0, 0.0
                else:
                    na, nb, k, ek, ed = truncate(w[a], w[b], rule_tau, rule_k)
                delta = (nb @ na).to(orig[key].dtype)
                sd[key].copy_(orig[key] + delta.to(sd[key].device))
                keep += ek; drop += ed; ktot += k; ntot += w[a].shape[0]
            L = math.sqrt(drop / (keep + drop)) if keep + drop > 0 else 0.0
            return L, ktot / max(ntot, 1)

        def restore():
            sd = tf.state_dict()
            for k, v in orig.items():
                sd[k].copy_(v)

        restore()
        m_base = generate(dom["prompts"], dom["probe"])
        apply(None, None)
        m_full = generate(dom["prompts"], dom["probe"])
        rec = {"adapter": name, "probe": dom["probe"], "n_prompts": len(dom["prompts"]),
               "metric_base": m_base, "metric_orig": m_full,
               "headroom": m_full - m_base, "variants": {}}
        print(f"{name:11s} base={m_base:.4f} full={m_full:.4f} head={m_full-m_base:+.4f}",
              flush=True)

        if not args.pilot:
            for tau in args.thresholds:
                L, frac = apply(tau, None)
                m = generate(dom["prompts"], dom["probe"])
                hd = m_full - m_base
                rec["variants"][f"e{round(tau*100):02d}"] = {
                    "L_W": L, "rank_frac": frac, "metric_value": m,
                    "retained": (m - m_base) / hd if hd > 0 else float("nan")}
                v = rec["variants"][f"e{round(tau*100):02d}"]
                print(f"  e{round(tau*100):02d}: L_W={L:.3f} rank={frac:.2f} "
                      f"m={m:.4f} u={v['retained']:+.3f}", flush=True)
        restore()
        results.append(rec)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(results)} adapters)")


if __name__ == "__main__":
    main()
