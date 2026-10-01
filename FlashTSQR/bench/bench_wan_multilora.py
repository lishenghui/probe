"""LoRAForge feasibility study, part 1: the EXECUTION side.

Question this answers: on a real video-diffusion DiT (Wan 2.1 T2V-1.3B), how
much does request-level multi-LoRA composition actually cost per denoising
step, and how much of that does fusion give back?

The three numbers we are after (see also bench_wan_merge.py for T_merge):

  (1) multi-LoRA overhead      (T_3lora - T_base) / T_base
  (2) fusion recovery          (T_3lora - T_fused) / T_3lora
  (3) break-even steps  S*  =  T_merge / (T_3lora - T_fused)

Variants timed (one full transformer forward == one denoising step):

  base        no adapters
  lora1       one rank-32 adapter, separate branch
  lora3       Disco-LoRA composition: content r32 + style r32 + motion r64,
              three separate branches (what diffusers/peft do today)
  stack128    the three stacked into ONE rank-128 branch (exact fusion,
              factored form -- no quality loss, this is the free win)
  comp64/48/32  fused adapter recompressed to a lower rank (FraQ-style)
  dense       Delta W folded into W0 (upper bound: base compute, but costs a
              full private copy of the 1.3B weights per composition)

Weights are randomly initialised from the published Wan 2.1 1.3B config.
Latency does not depend on weight *values*; quality does, and quality is
explicitly out of scope here (part 3).
"""

import argparse
import json
import os
import statistics
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- model config

# Wan-AI/Wan2.1-T2V-1.3B-Diffusers  transformer/config.json, verbatim.
WAN_1_3B = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=12,
    attention_head_dim=128,      # -> inner_dim = 1536
    in_channels=16,
    out_channels=16,
    text_dim=4096,
    freq_dim=256,
    ffn_dim=8960,
    num_layers=30,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    image_dim=None,
    added_kv_proj_dim=None,
    rope_max_seq_len=1024,
)

# The modules community Wan LoRAs actually target (all of attn + ffn).
TARGETS = (
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ffn.net.0.proj", "ffn.net.2",
)

# Disco-LoRA: content r32 + style r32 + motion r64.
DISCO_RANKS = (32, 32, 64)


# ------------------------------------------------------------------ lora layer

class LoRALinear(nn.Module):
    """A base Linear plus `len(ranks)` independent LoRA branches.

    ranks=[32,32,64] reproduces today's multi-adapter execution; ranks=[128] is
    the exactly-fused stack; ranks=[32] a recompressed fusion; ranks=[] is base.
    """

    def __init__(self, base: nn.Linear, ranks, scale=1.0):
        super().__init__()
        self.base = base
        self.scale = scale
        dev, dt = base.weight.device, base.weight.dtype
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        for r in ranks:
            a = torch.randn(r, d_in, device=dev, dtype=dt) * (1.0 / d_in) ** 0.5
            b = torch.randn(d_out, r, device=dev, dtype=dt) * (1.0 / r) ** 0.5
            self.A.append(nn.Parameter(a, requires_grad=False))
            self.B.append(nn.Parameter(b, requires_grad=False))

    def forward(self, x):
        y = self.base(x)
        for a, b in zip(self.A, self.B):
            y = y + self.scale * F.linear(F.linear(x, a), b)
        return y


def _split(model, path):
    parent = model
    parts = path.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]


def inject(model, ranks):
    """Wrap every target Linear with `ranks` LoRA branches. Returns count."""
    n = 0
    for i, block in enumerate(model.blocks):
        for t in TARGETS:
            parent, leaf = _split(block, t)
            base = getattr(parent, leaf)
            assert isinstance(base, nn.Linear), (i, t, type(base))
            setattr(parent, leaf, LoRALinear(base, ranks))
            n += 1
    return n


def strip(model):
    """Undo inject(), restoring the plain Linears."""
    for block in model.blocks:
        for t in TARGETS:
            parent, leaf = _split(block, t)
            m = getattr(parent, leaf)
            if isinstance(m, LoRALinear):
                setattr(parent, leaf, m.base)


@torch.no_grad()
def fold_dense(model, ranks, scale=1.0):
    """Fold sum_i B_i A_i straight into W0 (the 'dense merge' variant)."""
    n = 0
    for block in model.blocks:
        for t in TARGETS:
            parent, leaf = _split(block, t)
            base = getattr(parent, leaf)
            d_in, d_out = base.in_features, base.out_features
            dev, dt = base.weight.device, base.weight.dtype
            for r in ranks:
                a = torch.randn(r, d_in, device=dev, dtype=dt) * (1.0 / d_in) ** 0.5
                b = torch.randn(d_out, r, device=dev, dtype=dt) * (1.0 / r) ** 0.5
                base.weight.data.add_(scale * (b @ a))
            n += 1
    return n


# --------------------------------------------------------------------- harness

def make_inputs(batch, frames_lat, h_lat, w_lat, dtype, dev, text_len=512):
    hs = torch.randn(batch, 16, frames_lat, h_lat, w_lat, device=dev, dtype=dtype)
    eh = torch.randn(batch, text_len, WAN_1_3B["text_dim"], device=dev, dtype=dtype)
    ts = torch.full((batch,), 500, device=dev, dtype=torch.long)
    return hs, ts, eh


@torch.inference_mode()
def timeit(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0


@torch.inference_mode()
def count_kernels(fn):
    """Number of GPU kernel/memcpy/memset launches in one call, via the trace."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            tr = json.load(f)
        ev = tr["traceEvents"] if isinstance(tr, dict) else tr
        cats = ("kernel", "gpu_memcpy", "gpu_memset")
        return sum(1 for e in ev if e.get("ph") == "X" and e.get("cat") in cats)
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ micro test

@torch.inference_mode()
def micro(seq, dtype, dev):
    """Isolated projection: how much do the LoRA branches cost on one Linear?"""
    print("\n### micro: one projection, seq =", seq)
    print(f"{'shape':>14} | {'base':>8} | {'+1x r32':>8} | {'+3x lora':>9} | "
          f"{'+stack128':>9} | {'+r32':>8} | {'3lora/base':>10} | {'stack/3lora':>11}")
    for d_in, d_out, tag in ((1536, 1536, "attn 1536^2"),
                             (1536, 8960, "ffn up"),
                             (8960, 1536, "ffn down")):
        x = torch.randn(1, seq, d_in, device=dev, dtype=dtype)
        base = nn.Linear(d_in, d_out, device=dev, dtype=dtype)
        cfgs = {"base": [], "l1": [32], "l3": list(DISCO_RANKS), "st": [128], "c32": [32]}
        res = {}
        for k, ranks in cfgs.items():
            m = LoRALinear(base, ranks)
            res[k], _ = timeit(lambda: m(x))
            del m
        print(f"{tag:>14} | {res['base']:8.3f} | {res['l1']:8.3f} | {res['l3']:9.3f} | "
              f"{res['st']:9.3f} | {res['c32']:8.3f} | "
              f"{res['l3'] / res['base']:9.3f}x | "
              f"{(res['l3'] - res['st']) / res['l3'] * 100:10.1f}%")


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=49, help="video frames (pixel space)")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50, help="denoising steps for the extrapolation")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--layers", type=int, default=None, help="override num_layers (smoke test)")
    ap.add_argument("--no-kernels", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)

    cfg = dict(WAN_1_3B)
    if args.layers:
        cfg["num_layers"] = args.layers

    # latent geometry: VAE is 8x spatial, 4x temporal (+1 keyframe)
    f_lat = (args.frames - 1) // 4 + 1
    h_lat, w_lat = args.height // 8, args.width // 8
    p_t, p_h, p_w = cfg["patch_size"]
    seq = (f_lat // p_t) * (h_lat // p_h) * (w_lat // p_w)
    print(f"video {args.frames}f {args.height}x{args.width} -> latent "
          f"[{f_lat},{h_lat},{w_lat}] -> {seq} tokens, {cfg['num_layers']} layers, "
          f"batch {args.batch}")

    from diffusers import WanTransformer3DModel

    print("building model ...", flush=True)
    with torch.device(dev):
        model = WanTransformer3DModel(**cfg)
    model = model.to(dtype).eval().requires_grad_(False)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"transformer params: {nparam / 1e9:.3f}B")

    hs, ts, eh = make_inputs(args.batch, f_lat, h_lat, w_lat, dtype, dev)

    def fwd():
        return model(hidden_states=hs, timestep=ts, encoder_hidden_states=eh,
                     return_dict=False)[0]

    micro(seq, dtype, dev)

    variants = [
        ("base",       []),
        ("lora1_r32",  [32]),
        ("lora3_disco", list(DISCO_RANKS)),
        ("stack_r128", [128]),
        ("comp_r64",   [64]),
        ("comp_r48",   [48]),
        ("comp_r32",   [32]),
        ("dense_merge", None),     # special-cased
    ]

    print("\n### end-to-end: one denoising step (Wan 2.1 1.3B DiT)")
    hdr = (f"{'variant':>12} | {'ms/step':>9} | {'sd':>6} | {'x base':>7} | "
           f"{f'{args.steps}-step s':>13} | {'peak GiB':>9} | {'kernels':>8}")
    print(hdr)
    print("-" * len(hdr))

    out = {"config": vars(args), "seq": seq, "params": nparam, "variants": {}}
    for name, ranks in variants:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if name == "dense_merge":
            nmod = fold_dense(model, list(DISCO_RANKS))
        else:
            nmod = inject(model, ranks)
        ms, sd = timeit(fwd, reps=args.reps)
        peak = torch.cuda.max_memory_allocated() / 2**30
        nk = -1 if args.no_kernels else count_kernels(fwd)
        base_ms = out["variants"].get("base", {}).get("ms", ms)
        print(f"{name:>12} | {ms:9.2f} | {sd:6.2f} | {ms / base_ms:6.3f}x | "
              f"{ms * args.steps / 1000:13.2f} | {peak:9.2f} | {nk:8d}")
        out["variants"][name] = dict(ms=ms, sd=sd, peak_gib=peak, kernels=nk,
                                     modules=nmod, ranks=ranks)
        if name != "dense_merge":
            strip(model)
        else:
            break   # weights are dirty now; dense is the last variant

    # ---------------------------------------------------------------- verdict
    v = out["variants"]
    b, l3 = v["base"]["ms"], v["lora3_disco"]["ms"]
    print("\n### the numbers that decide the project")
    print(f"(1) multi-LoRA overhead   (T_3lora - T_base)/T_base       = "
          f"{(l3 - b) / b * 100:6.2f} %")
    for k in ("stack_r128", "comp_r64", "comp_r48", "comp_r32", "dense_merge"):
        if k in v:
            f = v[k]["ms"]
            gain = (l3 - f) / l3 * 100
            print(f"(2) fusion recovery via {k:<11} (T_3lora-T_f)/T_3lora = "
                  f"{gain:6.2f} %   ({(l3 - f):.2f} ms/step, "
                  f"{(l3 - f) * args.steps / 1000:.2f} s over {args.steps} steps)")
    print("\n(3) break-even S* = T_merge / (T_3lora - T_fused): needs T_merge "
          "from bench_wan_merge.py")
    for k in ("stack_r128", "comp_r64", "comp_r32"):
        if k in v:
            d = l3 - v[k]["ms"]
            if d > 0:
                print(f"    {k:<11}: saves {d:.2f} ms/step -> pays back a merge of "
                      f"T_merge ms after S* = T_merge/{d:.2f} steps")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
