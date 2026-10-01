#!/usr/bin/env python3
"""Zero-pad a variable-rank PEFT adapter to kernel-friendly rank buckets."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def next_power_of_two(value: int, maximum: int) -> int:
    return min(maximum, 1 << (value - 1).bit_length())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maximum", type=int, default=32)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source_weights = args.input / "adapter_model.safetensors"
    tensors = load_file(source_weights)
    with safe_open(source_weights, framework="pt") as f:
        metadata = f.metadata()
    config = json.loads((args.input / "adapter_config.json").read_text())
    rank_pattern = config.get("rank_pattern", {})
    new_pattern = dict(rank_pattern)
    report = []

    for a_name in sorted(k for k in tensors if k.endswith(".lora_A.weight")):
        b_name = a_name.replace(".lora_A.weight", ".lora_B.weight")
        a, b = tensors[a_name], tensors[b_name]
        old_rank = a.shape[0]
        new_rank = next_power_of_two(old_rank, args.maximum)
        if new_rank != old_rank:
            padded_a = torch.zeros((new_rank, a.shape[1]), dtype=a.dtype)
            padded_b = torch.zeros((b.shape[0], new_rank), dtype=b.dtype)
            padded_a[:old_rank].copy_(a)
            padded_b[:, :old_rank].copy_(b)
            tensors[a_name], tensors[b_name] = padded_a, padded_b
        module_name = a_name.removesuffix(".lora_A.weight")
        matches = [key for key in rank_pattern if module_name.endswith(key)]
        if len(matches) != 1:
            raise RuntimeError(f"cannot map tensor module to rank_pattern: {module_name}")
        new_pattern[matches[0]] = new_rank
        report.append({"module": matches[0], "old_rank": old_rank, "new_rank": new_rank})

    config["rank_pattern"] = new_pattern
    save_file(tensors, args.output / "adapter_model.safetensors", metadata=metadata)
    (args.output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {
        "source": str(args.input),
        "layers": len(report),
        "old_rank_sum": sum(x["old_rank"] for x in report),
        "new_rank_sum": sum(x["new_rank"] for x in report),
        "buckets": sorted(set(x["new_rank"] for x in report)),
        "layers_detail": report,
    }
    (args.output / "bucketing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "layers_detail"}, indent=2))


if __name__ == "__main__":
    main()
