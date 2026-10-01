#!/usr/bin/env python3
"""How much of each adapted layer does the LoRA update actually account for?

Perturbing the update by L_W = 0.30 moved a Qwen3-VL layer's output by only
0.45%, because the update is a small correction on top of a much larger base
weight.  If that ratio is what sets the entry-point perturbation, then the VLA's
undamped response should show up here as a far stronger adapter, and the
difference between task families needs no appeal to execution dynamics at all.

    S = || scale * B A ||_F / || W ||_F
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def factors(a, b):
    r = a.shape[0]
    a2 = a.reshape(r, -1).float().to(DEVICE)
    if b.ndim == 2:
        b2 = b.float()
    else:
        axis = next(i for i, s in enumerate(b.shape) if s == r and i > 0)
        b2 = b.movedim(axis, -1).reshape(-1, r).float()
    return a2, b2.to(DEVICE)


def product_norm(a, b):
    a2, b2 = factors(a, b)
    return float(((a2 @ a2.T) * (b2.T @ b2)).sum().clamp_min(0).sqrt())


def base_name(a_key: str) -> str:
    """PEFT decorates the module path; recover the base checkpoint's key."""
    name = re.sub(r"\.lora_[AB]\.(weight|default\.weight)$", ".weight", a_key)
    name = re.sub(r"\.lora_(down|up)\.weight$", ".weight", name)
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tail-half", action="store_true",
                        help="correction adapters store cat(truncated, original); "
                             "the tail alone is the original update")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg_path = args.adapter / "adapter_config.json"
    scale = 1.0
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        r, alpha = float(cfg["r"]), float(cfg["lora_alpha"])
        scale = alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)

    weights = args.adapter if args.adapter.is_file() else next(args.adapter.glob("*.safetensors"))
    pairs = {}
    with safe_open(weights, framework="pt", device="cpu") as h:
        keys = set(h.keys())
        for a_key in sorted(k for k in keys if ".lora_A." in k or ".lora_down." in k):
            b_key = a_key.replace(".lora_A.", ".lora_B.").replace(".lora_down.", ".lora_up.")
            if b_key in keys:
                pairs[a_key] = (h.get_tensor(a_key), h.get_tensor(b_key))

    # Base weight norms, streamed shard by shard.
    wanted = {base_name(k): k for k in pairs}
    norms = {}
    shards = sorted(args.base.glob("*.safetensors"))
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as h:
            for key in h.keys():
                if key in wanted:
                    norms[key] = float(h.get_tensor(key).float().norm())
    print(f"{args.family}: {len(pairs)} adapted modules, matched {len(norms)} base weights",
          flush=True)

    ratios = []
    for a_key, (a, b) in pairs.items():
        w = norms.get(base_name(a_key))
        if not w:
            continue
        a2, b2 = factors(a, b)
        if args.tail_half:
            half = a2.shape[0] // 2
            a2, b2 = a2[half:], b2[:, half:]
        delta = float(((a2 @ a2.T) * (b2.T @ b2)).sum().clamp_min(0).sqrt()) * scale
        ratios.append(delta / w)

    if not ratios:
        raise SystemExit("no adapter/base pairs matched -- check name mapping")
    ratios.sort()
    n = len(ratios)
    stat = {"family": args.family, "modules": n, "scale": scale,
            "mean": sum(ratios) / n, "median": ratios[n // 2],
            "p10": ratios[n // 10], "p90": ratios[int(0.9 * n)], "max": ratios[-1]}
    print(f"  S = ||dW||/||W||   mean={stat['mean']:.4f} median={stat['median']:.4f} "
          f"p10={stat['p10']:.4f} p90={stat['p90']:.4f} max={stat['max']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(args.output.read_text()) if args.output.exists() else []
    prev = [r for r in prev if r["family"] != args.family] + [stat]
    args.output.write_text(json.dumps(prev, indent=2) + "\n")


if __name__ == "__main__":
    main()
