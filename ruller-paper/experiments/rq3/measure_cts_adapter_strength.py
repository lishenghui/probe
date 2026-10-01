#!/usr/bin/env python3
"""Adapter strength for the released Compress-then-Serve LoRA collection.

That work reports LoRAs staying accurate under aggressive rank truncation and
uses this to serve thousands of them.  Our measurements suggest compression
tolerance tracks how much the adapter contributes to the adapted model, and the
one family that degrades (OpenVLA, -11.7% task success at e90) has
S = ||dW|| / ||W|| around 0.136 against 0.012-0.019 for the families that do
not degrade.

So the question is not whether their result reproduces, but which regime it sits
in.  Same estimator as measure_adapter_strength.py, applied to their released
adapters against the Mistral base they were trained on.
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


def base_name(a_key: str) -> str:
    name = re.sub(r"\.lora_[AB]\.(weight|default\.weight)$", ".weight", a_key)
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def product_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    a2 = a.reshape(a.shape[0], -1).float().to(DEVICE)
    b2 = b.reshape(-1, b.shape[-1]).float().to(DEVICE)
    return float(((a2 @ a2.T) * (b2.T @ b2)).sum().clamp_min(0).sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True,
                        help="directory of downloaded adapter repos")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Base weight norms are shared across every adapter, so pay for them once.
    norms: dict[str, float] = {}
    shards = sorted(args.base.rglob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors under {args.base}; found: "
                         f"{[p.name for p in args.base.iterdir()][:20]}")
    print(f"base shards: {[s.name for s in shards]}", flush=True)
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as h:
            for key in h.keys():
                if any(t in key for t in ("q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj")):
                    norms[key] = float(h.get_tensor(key).float().norm())
    print(f"base: {len(norms)} candidate weights", flush=True)

    rows = []
    repos = sorted(p for p in args.adapters.iterdir() if (p / "adapter_config.json").exists())
    for repo in repos:
        cfg = json.loads((repo / "adapter_config.json").read_text())
        r, alpha = float(cfg["r"]), float(cfg["lora_alpha"])
        scale = alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)
        weights = next(repo.glob("*.safetensors"))
        ratios = []
        with safe_open(weights, framework="pt", device="cpu") as h:
            keys = set(h.keys())
            for a_key in sorted(k for k in keys if ".lora_A." in k):
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                if b_key not in keys:
                    continue
                w = norms.get(base_name(a_key))
                if not w:
                    continue
                ratios.append(scale * product_norm(h.get_tensor(a_key), h.get_tensor(b_key)) / w)
        if not ratios:
            print(f"  {repo.name}: no matched modules", flush=True)
            continue
        ratios.sort()
        rows.append({"adapter": repo.name, "rank": int(r), "scale": scale,
                     "modules": len(ratios), "mean": sum(ratios) / len(ratios),
                     "median": ratios[len(ratios) // 2], "max": ratios[-1]})
        print(f"  {repo.name:52s} r={int(r):3d} modules={len(ratios):3d} "
              f"S_mean={rows[-1]['mean']:.4f}", flush=True)

    if not rows:
        raise SystemExit("no adapters matched the base weights")
    means = sorted(x["mean"] for x in rows)
    n = len(means)
    print(f"\n{n} adapters:  S_mean median={means[n//2]:.4f}  "
          f"p10={means[n//10]:.4f}  p90={means[int(0.9*n)]:.4f}  "
          f"min={means[0]:.4f}  max={means[-1]:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
