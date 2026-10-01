#!/usr/bin/env python3
"""Prediction-space divergence for the Wan2.1-T2V video pool under truncation.

The task-metric route on this pool is closed: a CLIP text probe cannot see what
these adapters do. The diagnostic is unambiguous -- at a fixed prompt and seed the
full PanoWan adapter changes 48.6% of the video in relative pixel norm while the
CLIP alignment moves by -0.009 with a standard deviation of 0.015 over prompts.
The adapter acts; the proxy metric is blind to it, so retained utility has no
usable denominator.

Divergence needs no proxy. Feed the same latent and timestep to the uncompressed
and the truncated adapter and compare the transformer's prediction,

    D_v(t) = || v_c(x_t, t) - v_o(x_t, t) || / || v_o(x_t, t) ||,

which is the quantity the four-family table of Sec. 4.5 already reports, but here
over a pool with an 8.7x strength spread and a full threshold sweep rather than
n = 4 at a single tau.

Only the transformer is loaded: no VAE, no text encoder, no sampling loop. The
text stream is random at a fixed seed, which is legitimate because every variant
sees the identical input and the comparison is between adapters, not against a
ground-truth video.

The pool's ranks are not uniform (16/32/64/128), so there is no comparable
fixed-rank column and only the threshold sweep is reported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_task_metrics import b_key, load_lora, to_base, truncate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.99, 0.95, 0.90, 0.80, 0.70, 0.50])
    ap.add_argument("--timesteps", type=int, nargs="+", default=[999, 750, 500, 250, 50])
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from diffusers import WanTransformer3DModel

    tf = WanTransformer3DModel.from_pretrained(
        args.base, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda").eval()
    cfg = tf.config
    print(f"transformer loaded: {cfg.num_layers} layers, in_channels={cfg.in_channels}",
          flush=True)

    # VAE stride is 8 spatially and 4 temporally for Wan2.1
    lat_f = (args.frames - 1) // 4 + 1
    lat_h, lat_w = args.height // 8, args.width // 8

    def inputs(rep: int):
        g = torch.Generator(device="cuda").manual_seed(args.seed + rep)
        hs = torch.randn(1, cfg.in_channels, lat_f, lat_h, lat_w, generator=g,
                         device="cuda", dtype=torch.bfloat16)
        eh = torch.randn(1, 512, cfg.text_dim, generator=g, device="cuda",
                         dtype=torch.bfloat16)
        return hs, eh

    pool = json.loads(args.pool.read_text())
    names = (args.adapters or sorted(pool))[args.shard::args.shards]
    print(f"shard {args.shard}/{args.shards}: {len(names)} adapters", flush=True)
    # the per-threshold cost is 300 QR+SVD factorisations; on CPU that dominated
    # the wall clock for the rank-128 adapter, so the factors move to the GPU once
    orig = {k: v.detach().clone() for k, v in tf.state_dict().items()}

    def restore():
        sd = tf.state_dict()
        for k, v in orig.items():
            sd[k].copy_(v)

    def apply(pairs, w, tau):
        """tau=None applies the uncompressed adapter. Returns (L_W, rank fraction)."""
        sd = tf.state_dict()
        keep = drop = 0.0
        ktot = ntot = 0
        for a, b, tgt in pairs:
            key = tgt + ".weight"
            if tau is None:
                na, nb = w[a].float(), w[b].float()
                k, ek, ed = na.shape[0], 0.0, 0.0
            else:
                na, nb, k, ek, ed = truncate(w[a], w[b], tau, None)
            delta = (nb @ na).to(device=orig[key].device, dtype=orig[key].dtype)
            sd[key].copy_(orig[key] + delta)
            keep += ek
            drop += ed
            ktot += k
            ntot += w[a].shape[0]
        L = math.sqrt(drop / (keep + drop)) if keep + drop > 0 else 0.0
        return L, ktot / max(ntot, 1)

    @torch.no_grad()
    def predict(rep, ts):
        hs, eh = inputs(rep)
        t = torch.tensor([ts], device="cuda", dtype=torch.long)
        return tf(hidden_states=hs, timestep=t, encoder_hidden_states=eh,
                  return_dict=False)[0].float()

    results = []
    for name in names:
        w = {k: v.to("cuda") for k, v in load_lora(Path(pool[name]["path"])).items()}
        # carry every scalar the pool file supplies, so a grouping key added later
        # (ladder membership, uploader) is not silently dropped from the results
        meta = {k: v for k, v in pool[name].items() if k != "path"}
        pairs = []
        for a in sorted(k for k in w if ".lora_A" in k or ".lora_down" in k):
            b = b_key(a)
            tgt = to_base(a)
            if b in w and tgt + ".weight" in orig:
                pairs.append((a, b, tgt))
        restore()
        apply(pairs, w, None)
        ref = {(r, t): predict(r, t) for r in range(args.repeats) for t in args.timesteps}
        num = sum(float(v.norm()) for v in ref.values())
        print(f"\n{name}: {len(pairs)} modules, reference built "
              f"(mean |v_o| = {num/len(ref):.2f})", flush=True)

        rec = {"adapter": name, "modules": len(pairs), **meta, "variants": {}}
        for tau in args.thresholds:
            restore()
            L, frac = apply(pairs, w, tau)
            vals = []
            for r in range(args.repeats):
                for t in args.timesteps:
                    got = predict(r, t)
                    o = ref[(r, t)]
                    vals.append(float((got - o).norm() / o.norm().clamp_min(1e-8)))
            lab = f"e{round(tau * 100):02d}"
            rec["variants"][lab] = {
                "L_W": L, "rank_frac": frac,
                "d_v_mean": sum(vals) / len(vals),
                "d_v_max": max(vals), "n": len(vals)}
            v = rec["variants"][lab]
            print(f"  {lab}: L_W={L:.3f} rank={frac:.2f} d_v={v['d_v_mean']:.5f}",
                  flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results + [rec], indent=2) + "\n")
        restore()
        results.append(rec)
    print(f"\nwrote {args.output} ({len(results)} adapters)")


if __name__ == "__main__":
    main()
