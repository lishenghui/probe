#!/usr/bin/env python3
"""Create behavior-equivalent, energy-truncated PEFT LoRA adapters.

The output tensors retain the original nominal shape and scaling, but only the
first k factor coordinates are populated.  This is exactly equivalent to a
repacked rank-k adapter and avoids PEFT rank_pattern compatibility differences
during behavioral evaluation.  Non-LoRA tensors are copied bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.95, 0.90])
    return parser.parse_args()


def load_tensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return tensors, metadata


def b_key_for(a_key: str) -> str:
    if ".lora_A." in a_key:
        return a_key.replace(".lora_A.", ".lora_B.")
    if ".lora_down." in a_key:
        return a_key.replace(".lora_down.", ".lora_up.")
    raise ValueError(f"Unsupported LoRA A/down key: {a_key}")


def product_frob_sq(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute ||B A||_F^2 without materializing the dense update."""
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], b.shape[1])
    a32, b32 = a.float(), b.float()
    gram_a = a32 @ a32.T
    gram_b = b32.T @ b32
    return float((gram_a * gram_b.T).sum().clamp_min(0).item())


def truncated_factors(
    a: torch.Tensor, b: torch.Tensor, threshold: float, fixed_rank: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Return zero-padded factors whose product is truncated-SVD(B @ A).

    `fixed_rank` overrides the energy criterion and keeps exactly that many
    directions, which is the other compression rule in common use: every adapter
    is given the same rank rather than the same retained energy.
    """
    original_dtype = a.dtype
    original_a_shape, original_b_shape = a.shape, b.shape
    # PEFT Conv2d LoRA uses A=[r,in,k,k] and B=[out,r,1,1].  Its effective
    # update is the same matrix product after flattening A's input/kernel axes
    # and B's singleton spatial axes.
    if a.ndim > 2:
        if b.ndim != a.ndim or any(size != 1 for size in b.shape[2:]):
            raise ValueError(f"Unsupported convolutional LoRA A={tuple(a.shape)}, B={tuple(b.shape)}")
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], b.shape[1])
    a32, b32 = a.float(), b.float()
    rank = a32.shape[0]
    if b32.shape[1] != rank:
        raise ValueError(f"Incompatible shapes A={tuple(a.shape)}, B={tuple(b.shape)}")

    # Thin QR reduces the expensive SVD to at most rank x rank.
    qb, rb = torch.linalg.qr(b32, mode="reduced")
    qa, ra = torch.linalg.qr(a32.T, mode="reduced")
    uc, singular, vhc = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    energy = singular.square()
    total = energy.sum()
    if fixed_rank is not None:
        kept = max(0, min(int(fixed_rank), int(singular.numel())))
    elif total <= 0:
        kept = 0
    else:
        cumulative = torch.cumsum(energy, dim=0) / total
        kept = int(torch.searchsorted(cumulative, threshold).item()) + 1

    new_a = torch.zeros_like(a32)
    new_b = torch.zeros_like(b32)
    if kept:
        root_s = singular[:kept].sqrt()
        new_b[:, :kept] = (qb @ uc[:, :kept]) * root_s.unsqueeze(0)
        new_a[:kept, :] = root_s.unsqueeze(1) * (vhc[:kept, :] @ qa.T)
    retained = float(energy[:kept].sum() / total) if total > 0 else 1.0
    return (
        new_a.reshape(original_a_shape).to(original_dtype),
        new_b.reshape(original_b_shape).to(b.dtype),
        kept,
        retained,
    )


def main() -> None:
    args = parse_args()
    tensors, metadata = load_tensors(args.weights)
    a_keys = sorted(
        key for key in tensors if ".lora_A." in key or ".lora_down." in key
    )
    if not a_keys:
        raise RuntimeError("No LoRA A/down tensors found")
    pairs = [(a_key, b_key_for(a_key)) for a_key in a_keys]
    missing = [b_key for _, b_key in pairs if b_key not in tensors]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} B/up tensors; first: {missing[0]}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "source_weights": str(args.weights.resolve()),
        "source_config": str(args.config.resolve()),
        "pair_count": len(pairs),
        "non_lora_tensors_preserved": len(tensors) - 2 * len(pairs),
        "variants": {},
    }
    total_nominal = sum(
        tensors[a].shape[0] * (tensors[a].shape[1] + tensors[b].shape[0])
        for a, b in pairs
    )

    for threshold in args.thresholds:
        label = f"e{round(threshold * 100):02d}"
        out_dir = args.output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        output = dict(tensors)
        layer_stats = []
        retained_params = 0
        total_update_sq = 0.0
        total_residual_sq = 0.0
        for a_key, b_key in pairs:
            new_a, new_b, kept, retained_energy = truncated_factors(
                tensors[a_key], tensors[b_key], threshold
            )
            output[a_key], output[b_key] = new_a, new_b
            update_sq = product_frob_sq(tensors[a_key], tensors[b_key])
            # B A - B_k A_k = [B, -B_k] [A; A_k].
            residual_a = torch.cat((tensors[a_key], new_a), dim=0)
            residual_b = torch.cat((tensors[b_key], -new_b), dim=1)
            residual_sq = product_frob_sq(residual_a, residual_b)
            total_update_sq += update_sq
            total_residual_sq += residual_sq
            in_features = tensors[a_key].shape[1]
            out_features = tensors[b_key].shape[0]
            retained_params += kept * (in_features + out_features)
            layer_stats.append(
                {
                    "module": a_key.rsplit(".lora_A.", 1)[0] if ".lora_A." in a_key else a_key.rsplit(".lora_down.", 1)[0],
                    "nominal_rank": tensors[a_key].shape[0],
                    "retained_rank": kept,
                    "retained_rank_ratio": kept / tensors[a_key].shape[0],
                    "retained_energy": retained_energy,
                    "L_W": math.sqrt(residual_sq / update_sq) if update_sq else 0.0,
                }
            )
        save_file(output, out_dir / "adapter_model.safetensors", metadata=metadata)
        shutil.copy2(args.config, out_dir / "adapter_config.json")
        variant = {
            "threshold": threshold,
            "layers": len(layer_stats),
            "mean_retained_rank": sum(x["retained_rank"] for x in layer_stats) / len(layer_stats),
            "min_retained_rank": min(x["retained_rank"] for x in layer_stats),
            "max_retained_rank": max(x["retained_rank"] for x in layer_stats),
            "mean_retained_rank_ratio": sum(x["retained_rank_ratio"] for x in layer_stats) / len(layer_stats),
            "median_retained_rank_ratio": sorted(x["retained_rank_ratio"] for x in layer_stats)[len(layer_stats) // 2],
            "lora_parameter_fraction_if_repacked": retained_params / total_nominal,
            "global_L_W": math.sqrt(total_residual_sq / total_update_sq) if total_update_sq else 0.0,
            "layer_stats": layer_stats,
        }
        (out_dir / "compression.json").write_text(json.dumps(variant, indent=2) + "\n")
        summary["variants"][label] = {k: v for k, v in variant.items() if k != "layer_stats"}
        print(label, json.dumps(summary["variants"][label]))

    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
