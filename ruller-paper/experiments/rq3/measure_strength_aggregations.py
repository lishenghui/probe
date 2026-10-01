#!/usr/bin/env python3
"""Is the strength-sensitivity result an artefact of how S is aggregated?

S is defined per module as ||alpha * B A||_F / ||W||_F, which is scale-free, but
an adapter touches hundreds of modules of different shapes and the per-adapter
number depends on how those are combined.  Three defensible choices disagree in
principle: an unweighted mean gives a 4096x4096 attention projection the same
vote as a small one; the global ratio ||alpha * dW||_F / ||W||_F over the
concatenation weights by parameter count; the median is robust to outlier
modules.  The conclusion has to survive all of them, so report all of them.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def frob_of_product(a: torch.Tensor, b: torch.Tensor) -> float:
    """||B A||_F via the Gram trick; forming B A would be 4096x11008."""
    a2 = a.reshape(a.shape[0], -1).float().to(DEVICE)
    b2 = b.reshape(b.shape[0], b.shape[1]).float().to(DEVICE)
    return float(((a2 @ a2.T) * (b2.T @ b2)).sum().clamp_min(0).sqrt())


def base_key(a_key: str) -> str:
    name = re.sub(r"\.lora_A\.(default\.)?weight$", ".weight", a_key)
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapters", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    norms = {}
    for shard in sorted(args.base.rglob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as h:
            for key in h.keys():
                if any(t in key for t in TARGETS):
                    norms[key] = float(h.get_tensor(key).float().norm())
    print(f"base: {len(norms)} candidate weights", flush=True)

    rows = []
    for repo in sorted(p for p in args.adapters.iterdir() if p.is_dir()):
        weights = next(repo.glob("*.safetensors"), None)
        cfg_path = repo / "adapter_config.json"
        if weights is None or not cfg_path.is_file():
            continue
        cfg = json.loads(cfg_path.read_text())
        rank, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
        scale = alpha / (math.sqrt(rank) if cfg.get("use_rslora") else rank)

        per_module, num, den = [], 0.0, 0.0
        with safe_open(weights, framework="pt", device="cpu") as h:
            keys = set(h.keys())
            for a_key in sorted(k for k in keys if ".lora_A." in k):
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                w = norms.get(base_key(a_key))
                if b_key not in keys or not w:
                    continue
                d = scale * frob_of_product(h.get_tensor(a_key), h.get_tensor(b_key))
                per_module.append(d / w)
                num += d * d
                den += w * w
        if not per_module:
            continue
        per_module.sort()
        n = len(per_module)
        rows.append({
            "adapter": repo.name, "rank": rank, "scale": scale, "modules": n,
            "S_mean": sum(per_module) / n,
            "S_median": per_module[n // 2],
            "S_global": math.sqrt(num / den),
            "S_max": per_module[-1],
            "S_p90": per_module[int(0.9 * n)],
        })
        r = rows[-1]
        print(f"{repo.name:14s} mean={r['S_mean']:.4f} median={r['S_median']:.4f} "
              f"global={r['S_global']:.4f} max={r['S_max']:.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(rows)} adapters)")


if __name__ == "__main__":
    main()
