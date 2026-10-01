#!/usr/bin/env python3
"""What would these adapters weigh if they were stored at the rank they use?

Energy truncation can zero the tail of A and B without shrinking the tensors.
An adapter compressed that way is numerically low-rank but still occupies the
full rank on disk and in GPU memory, so it delivers the accuracy/latency side of
compression and none of the capacity side.  This measures the effective rank per
module from the stored weights and reports both sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def effective_rank(a: torch.Tensor, tol: float = 1e-8) -> int:
    """Rows of A that are not identically zero; the tail is what truncation left.

    A is flattened first: a vision backbone contributes 4-D conv adapters shaped
    (r, C, kh, kw), and a row norm taken without flattening counts elements
    rather than rows.
    """
    return int((a.reshape(a.shape[0], -1).float().norm(dim=1) > tol).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapters", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/lora_true_size.json"))
    args = parser.parse_args()

    rows = []
    print(f"{'adapter':34s} {'modules':>7s} {'stored r':>8s} {'mean r':>7s} "
          f"{'on disk':>9s} {'true':>9s} {'shrink':>7s}")
    for path in args.adapters:
        f = path if path.is_file() else next(path.glob("*.safetensors"))
        stored_bytes = true_bytes = 0
        ranks, stored_rank = [], 0
        with safe_open(f, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            a_keys = [k for k in keys if k.endswith("lora_A.weight")]
            for ak in a_keys:
                bk = ak.replace("lora_A.weight", "lora_B.weight")
                if bk not in keys:
                    continue
                a = handle.get_tensor(ak)
                b = handle.get_tensor(bk)
                r = effective_rank(a)
                ranks.append(r)
                stored_rank = max(stored_rank, a.shape[0])
                # Per-rank cost: one row of A plus one column of B, whatever the
                # trailing dimensions are.
                per_rank = a.numel() // a.shape[0] + b.numel() // b.shape[-1]
                stored_bytes += a.shape[0] * per_rank * a.element_size()
                true_bytes += r * per_rank * a.element_size()
        if not ranks:
            print(f"{path.name:34s}  no LoRA pairs found")
            continue
        mean_r = sum(ranks) / len(ranks)
        print(f"{str(path)[-34:]:34s} {len(ranks):7d} {stored_rank:8d} {mean_r:7.1f} "
              f"{stored_bytes/2**20:8.0f}M {true_bytes/2**20:8.0f}M "
              f"{stored_bytes/max(true_bytes,1):6.2f}x")
        rows.append({"adapter": str(path), "modules": len(ranks), "stored_rank": stored_rank,
                     "mean_rank": mean_r, "min_rank": min(ranks), "max_rank": max(ranks),
                     "stored_mib": stored_bytes / 2**20, "true_mib": true_bytes / 2**20})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
