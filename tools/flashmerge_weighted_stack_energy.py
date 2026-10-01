#!/usr/bin/env python3
"""FRAQ-compress a weighted stack of AnimateDiff MotionLoRA checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from analyze_lora_fraq_spectrum import load_flash_extension


def unwrap(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def pairs(state: dict[str, torch.Tensor]) -> list[tuple[str, str, str]]:
    result = []
    for key in state:
        if key.endswith(".down.weight"):
            prefix = key[: -len(".down.weight")]
            up = prefix + ".up.weight"
        elif key.endswith(".lora_down.weight"):
            prefix = key[: -len(".lora_down.weight")]
            up = prefix + ".lora_up.weight"
        else:
            continue
        if up not in state:
            raise KeyError(f"Missing pair for {key}: {up}")
        result.append((prefix, key, up))
    if not result:
        raise RuntimeError("No MotionLoRA down/up pairs found")
    return sorted(result)


def retained_ranks(s: torch.Tensor, threshold: float, multiple: int = 1) -> torch.Tensor:
    """Smallest rank per module reaching `threshold` of the spectral energy.

    `multiple` rounds that rank up.  cuBLAS drops off its tensor-core kernels
    when a GEMM dimension is not a multiple of 8, and the sidecar's rank *is*
    that dimension, so an arbitrary retained rank can cost several times what an
    aligned one does.  Rounding up keeps more singular values, so accuracy only
    improves.  See docs/rank_proportional_lora_inference.md.
    """
    energy = s.square()
    cumulative = energy.cumsum(1) / energy.sum(1, keepdim=True)
    ranks = (cumulative < threshold).sum(1).add(1)
    if multiple > 1:
        ranks = ranks.add(multiple - 1).div(multiple, rounding_mode="floor").mul(multiple)
    return ranks.clamp_max(s.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, action="append", required=True)
    ap.add_argument("--weight", type=float, action="append", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--energy-thresholds", type=float, nargs="+", default=[.95, .9, .8, .7])
    ap.add_argument(
        "--rank-multiple", type=int, default=8,
        help="round every retained rank up to this multiple; 1 disables. "
             "Alignment is what keeps the compressed sidecar on cuBLAS's "
             "tensor-core path, and it only adds singular values.",
    )
    ap.add_argument("--rows-per-leaf", type=int, default=128)
    ap.add_argument("--threads-per-block", type=int, default=256)
    ap.add_argument("--flash-kernel", type=Path,
                    default=Path(__file__).resolve().parents[1] / "FlashTSQR/kernels/tsqr_full.cu")
    args = ap.parse_args()
    if len(args.checkpoint) != len(args.weight):
        raise ValueError("--checkpoint and --weight counts differ")
    if not torch.cuda.is_available():
        raise RuntimeError("FlashMerge/FRAQ requires CUDA")
    if any(w <= 0 for w in args.weight):
        raise ValueError("Stack weights must be positive")
    if any(not 0 < t <= 1 for t in args.energy_thresholds):
        raise ValueError("Energy thresholds must be in (0, 1]")

    states = [unwrap(p) for p in args.checkpoint]
    reference = pairs(states[0])
    reference_keys = [(a, b) for _, a, b in reference]
    for path, state in zip(args.checkpoint[1:], states[1:]):
        if [(a, b) for _, a, b in pairs(state)] != reference_keys:
            raise ValueError(f"LoRA module layout differs: {path}")

    # Concatenation represents sum_i alpha_i B_i A_i exactly.
    stacked = []
    for prefix, down_key, up_key in reference:
        downs, ups = [], []
        for state, alpha in zip(states, args.weight):
            scale = math.sqrt(alpha)
            downs.append(state[down_key].float() * scale)
            ups.append(state[up_key].float() * scale)
        down = torch.cat(downs, dim=0)
        up = torch.cat(ups, dim=1)
        stacked.append((prefix, down_key, up_key, down, up))

    buckets = defaultdict(list)
    for item in stacked:
        buckets[(tuple(item[3].shape), tuple(item[4].shape))].append(item)
    outputs = {t: {} for t in args.energy_thresholds}
    ranks = {t: {} for t in args.energy_thresholds}
    achieved = {t: [] for t in args.energy_thresholds}
    # FlashTSQR's shared-memory leaf kernel targets small/medium ranks.  As in
    # flashmerge_energy_truncate_lora.py, rank-256 stacks use batched cuSOLVER
    # QR while retaining the same FRAQ QR-core-SVD and reconstruction.
    use_flash = stacked[0][3].shape[0] < 192
    extension = load_flash_extension(args.flash_kernel) if use_flash else None
    started = time.perf_counter()

    for bucket_index, items in enumerate(buckets.values(), 1):
        a = torch.stack([x[3] for x in items]).cuda()
        b = torch.stack([x[4] for x in items]).cuda()
        rank0 = a.shape[1]
        rpl_b = max(rank0, min(args.rows_per_leaf, b.shape[1]))
        rpl_a = max(rank0, min(args.rows_per_leaf, a.shape[2]))
        if use_flash:
            rb = extension.tsqr_factor(b, rpl_b, args.threads_per_block)
            ra = extension.tsqr_factor(a.transpose(1, 2).contiguous(), rpl_a, args.threads_per_block)
        else:
            qb, rb = torch.linalg.qr(b, mode="reduced")
            qa, ra = torch.linalg.qr(a.transpose(1, 2), mode="reduced")
        u, singular, vh = torch.linalg.svd(rb @ ra.transpose(1, 2))
        for threshold in args.energy_thresholds:
            keep = retained_ranks(singular, threshold, args.rank_multiple)
            kmax = int(keep.max())
            root = singular[:, :kmax].sqrt()
            left = (u[:, :, :kmax] * root[:, None, :]).contiguous()
            right = (vh[:, :kmax, :].transpose(1, 2) * root[:, None, :]).contiguous()
            if use_flash:
                extension.tsqr_factor(b, rpl_b, args.threads_per_block)
                b2 = extension.tsqr_applyQ(left, args.threads_per_block).cpu()
                extension.tsqr_factor(a.transpose(1, 2).contiguous(), rpl_a, args.threads_per_block)
                a2 = extension.tsqr_applyQ(right, args.threads_per_block).transpose(1, 2).cpu()
            else:
                b2 = (qb @ left).cpu()
                a2 = (qa @ right).transpose(1, 2).cpu()
            total = singular.square().sum(1)
            for pos, (prefix, down_key, up_key, _, _) in enumerate(items):
                k = int(keep[pos])
                outputs[threshold][down_key] = a2[pos, :k].contiguous()
                outputs[threshold][up_key] = b2[pos, :, :k].contiguous()
                ranks[threshold][prefix] = k
                achieved[threshold].append(float(singular[pos, :k].square().sum() / total[pos]))
        print(f"[{bucket_index}/{len(buckets)}] modules={len(items)} stacked_rank={rank0}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": "FlashMerge/FRAQ", "operation": "weighted-stack-then-compress",
        "sources": [{"path": str(p), "weight": w} for p, w in zip(args.checkpoint, args.weight)],
        "modules": len(reference), "stacked_rank": int(stacked[0][3].shape[0]),
        "elapsed_seconds": elapsed, "variants": {},
    }
    for threshold, state in outputs.items():
        label = f"e{round(threshold * 100):02d}"
        destination = args.output_dir / f"motionlora_stack_fraq_{label}.ckpt"
        torch.save(state, destination)
        values = list(ranks[threshold].values())
        manifest["variants"][label] = {
            "path": str(destination), "bytes": destination.stat().st_size,
            "rank_min": min(values), "rank_mean": sum(values) / len(values), "rank_max": max(values),
            "achieved_energy_min": min(achieved[threshold]),
            "achieved_energy_mean": sum(achieved[threshold]) / len(achieved[threshold]),
            "rank_by_module": ranks[threshold],
        }
        print(f"saved {destination}", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
