#!/usr/bin/env python3
"""How much error does compression inject, before any dynamics act on it?

The functional outcomes of energy truncation differ enormously across task
families -- code generation and image-edit judging are flat at e90, video
fidelity is flat, closed-loop manipulation loses 11.7% success.  Any claim that
this is about *error propagation* rather than *error magnitude* needs the
injected error itself measured on a common scale first.

That common scale is the relative change of the LoRA update in weight space:

    L_W = || B_k A_k - B A ||_F / || B A ||_F

reported per module and aggregated.  It is the one quantity every modality
shares, it is what the truncation directly controls, and it costs nothing to
compute from the adapters on disk.

Two adapter layouts appear in this project and both are handled:
  * zero-padded truncated adapters (compress_adapter.py) -- compare against the
    uncompressed adapter of the same family;
  * correction adapters (make_lora_corrections.py) -- these already *are*
    B_k A_k - B A, because the base checkpoint carries the original merged in,
    so their own product is the numerator and only the denominator is needed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def b_key_for(a_key: str) -> str:
    for down, up in ((".lora_A.", ".lora_B."), (".lora_down.", ".lora_up.")):
        if down in a_key:
            return a_key.replace(down, up)
    raise ValueError(a_key)


def load_pairs(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    weights = path if path.is_file() else next(path.glob("*.safetensors"))
    pairs = {}
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for a_key in sorted(k for k in keys if ".lora_A." in k or ".lora_down." in k):
            b_key = b_key_for(a_key)
            if b_key in keys:
                pairs[a_key] = (handle.get_tensor(a_key), handle.get_tensor(b_key))
    return pairs


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def factors(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise to (r, K) and (N, r).

    A conv adapter stores B as (out_channels, r, kh, kw), so the rank axis is 1
    rather than the last one; slicing the last axis there silently produces an
    empty tensor.  Normalising once up front means every caller can treat the
    rank as axis 0 of A and axis 1 of B.
    """
    r = a.shape[0]
    a2 = a.reshape(r, -1).float().to(DEVICE)
    if b.ndim == 2:
        b2 = b.float()
    else:
        axis = next(i for i, size in enumerate(b.shape) if size == r and i > 0)
        b2 = b.movedim(axis, -1).reshape(-1, r).float()
    return a2, b2.to(DEVICE)


def norm_of(a2: torch.Tensor, b2: torch.Tensor) -> float:
    """||B @ A||_F from already-normalised factors."""
    gram = (a2 @ a2.T) * (b2.T @ b2)
    return float(gram.sum().clamp_min(0).sqrt())


def product_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    """||B @ A||_F without forming B @ A.

    ||BA||_F^2 = tr(A^T B^T B A) = tr((A A^T)(B^T B)), and both factors are
    r x r.  Forming the full N x K product instead costs O(N K r), which for a
    5B-parameter adapter is minutes per module on CPU.
    """
    return norm_of(*factors(a, b))


def difference_norm(a1, b1, a2_, b2_) -> float:
    """||B1 A1 - B2 A2||_F, again via the r x r Gram trick on [B1 | -B2]."""
    a1f, b1f = factors(a1, b1)
    a2f, b2f = factors(a2_, b2_)
    return norm_of(torch.cat([a1f, a2f], dim=0), torch.cat([b1f, -b2f], dim=1))


def scaling(config: Path) -> float:
    cfg = json.loads(config.read_text())
    r, alpha = float(cfg["r"]), float(cfg["lora_alpha"])
    if cfg.get("use_rslora"):
        return alpha / math.sqrt(r)
    return alpha / r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--variant", type=Path, nargs="+", required=True)
    parser.add_argument("--correction", action="store_true",
                        help="variants are Delta_k - Delta rather than Delta_k")
    parser.add_argument("--correction-self-reference", action="store_true",
                        help="the correction also carries the negated original in its "
                             "second half, so both numerator and denominator come from it")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.correction_self_reference:
        # make_lora_corrections.py stores cat(truncated, original) and
        # cat(truncated, -original): the tail alone reconstructs the original
        # update, which the merged base checkpoint no longer exposes.
        base, denom = {}, {}
    else:
        base = load_pairs(args.original)
        base_scale = scaling(args.original / "adapter_config.json") \
            if (args.original / "adapter_config.json").exists() else 1.0
        denom = {k: base_scale * product_norm(*v) for k, v in base.items()}

    rows = []
    print(f"{'family':16s} {'variant':8s} {'modules':>7s} {'L_W mean':>9s} "
          f"{'median':>8s} {'p90':>8s} {'max':>8s}")
    for path in args.variant:
        pairs = load_pairs(path)
        var_scale = scaling(path / "adapter_config.json") \
            if (path / "adapter_config.json").exists() else 1.0
        ratios = []
        for key, (a, b) in pairs.items():
            if args.correction_self_reference:
                a2, b2 = factors(a, b)
                half = a2.shape[0] // 2
                num = var_scale * norm_of(a2, b2)
                # The tail is the negated original the merged checkpoint hides.
                den = var_scale * norm_of(a2[half:], b2[:, half:])
                if den > 0:
                    ratios.append(num / den)
                continue
            if key not in base or denom.get(key, 0.0) == 0.0:
                continue
            if args.correction:
                # The adapter already encodes the difference.
                num = var_scale * product_norm(a, b)
            else:
                ba, bb = base[key]
                num = difference_norm(a, var_scale * b, ba, base_scale * bb)
            ratios.append(num / denom[key])
        if not ratios:
            print(f"{args.family:16s} {path.name:8s}  no comparable modules")
            continue
        ratios.sort()
        n = len(ratios)
        stat = {"family": args.family, "variant": path.name, "modules": n,
                "mean": sum(ratios) / n, "median": ratios[n // 2],
                "p90": ratios[int(0.9 * n)], "max": ratios[-1]}
        rows.append(stat)
        print(f"{args.family:16s} {path.name:8s} {n:7d} {stat['mean']:9.4f} "
              f"{stat['median']:8.4f} {stat['p90']:8.4f} {stat['max']:8.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.output.read_text()) if args.output.exists() else []
    existing = [r for r in existing if r["family"] != args.family] + rows
    args.output.write_text(json.dumps(existing, indent=2) + "\n")


if __name__ == "__main__":
    main()
