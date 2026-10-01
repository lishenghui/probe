#!/usr/bin/env python3
"""How misaligned are the ranks in the FraQ adapters we have already produced?

Energy truncation picks whatever rank first reaches the target, which is an
arbitrary number.  cuBLAS drops off its tensor-core kernels when a GEMM
dimension is not a multiple of 8, and the sidecar's rank *is* that dimension --
measured on a Wan attention layer, a rank-241 sidecar costs 3.9x a rank-256 one
despite having 6% fewer flops.  This reports, per compressed variant, how much
of that exposure exists and what rounding every rank up to a multiple of 8 would
cost in extra rank rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aligned(r: int, multiple: int) -> bool:
    return r % multiple == 0


def analyse(ranks: dict[str, int], multiple: int) -> dict:
    values = list(ranks.values())
    total = len(values)
    rounded = [((r + multiple - 1) // multiple) * multiple for r in values]
    return {
        "modules": total,
        "distinct_ranks": len(set(values)),
        "aligned8": sum(1 for r in values if aligned(r, 8)),
        "aligned16": sum(1 for r in values if aligned(r, 16)),
        "rank_mean": sum(values) / total,
        "rank_mean_rounded": sum(rounded) / total,
        "extra_rank_pct": 100.0 * (sum(rounded) - sum(values)) / sum(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--multiple", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/kernel_bench/rank_alignment.json"))
    args = parser.parse_args()

    rows = []
    for manifest in sorted(args.root.rglob("*manifest*.json")):
        try:
            data = json.loads(manifest.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        variants = data.get("variants")
        if not isinstance(variants, dict):
            continue
        for label, info in variants.items():
            ranks = info.get("rank_by_module") if isinstance(info, dict) else None
            if not ranks:
                continue
            row = {"manifest": str(manifest.relative_to(args.root)), "variant": label}
            row.update(analyse(ranks, args.multiple))
            rows.append(row)

    if not rows:
        print(f"no manifests with rank_by_module under {args.root}")
        return

    seen = set()
    print(f"{'adapter':44s} {'var':5s} {'mods':>5s} {'distinct':>8s} "
          f"{'%8|r':>6s} {'%16|r':>6s} {'mean':>7s} {'->':>7s} {'+rank':>6s}")
    for row in rows:
        # One manifest per adapter is enough; reruns duplicate them.
        key = (Path(row["manifest"]).name, row["variant"])
        if key in seen:
            continue
        seen.add(key)
        name = Path(row["manifest"]).stem[:44]
        print(f"{name:44s} {row['variant']:5s} {row['modules']:5d} {row['distinct_ranks']:8d} "
              f"{100*row['aligned8']/row['modules']:5.0f}% {100*row['aligned16']/row['modules']:5.0f}% "
              f"{row['rank_mean']:7.1f} {row['rank_mean_rounded']:7.1f} "
              f"{row['extra_rank_pct']:5.1f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"multiple": args.multiple, "rows": rows}, indent=2) + "\n")
    kept = [r for r in rows]
    print(f"\n{len(seen)} compressed variants examined")
    print(f"median share of modules already aligned to {args.multiple}: "
          f"{sorted(100*r['aligned8']/r['modules'] for r in kept)[len(kept)//2]:.0f}%")
    print(f"median extra rank cost of aligning: "
          f"{sorted(r['extra_rank_pct'] for r in kept)[len(kept)//2]:.1f}%")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
