#!/usr/bin/env python3
"""Build correction LoRAs for a checkpoint that already has the source LoRA merged."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compress_adapter import b_key_for, truncated_factors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    with safe_open(args.weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    config = json.loads(args.config.read_text())
    source_rank = int(config["r"])
    source_alpha = float(config["lora_alpha"])
    a_keys = sorted(key for key in tensors if ".lora_A." in key)

    summary = {}
    for threshold in (0.99, 0.95, 0.90):
        label = f"e{round(threshold * 100):02d}"
        output = {}
        kept_ranks = []
        for a_key in a_keys:
            b_key = b_key_for(a_key)
            a, b = tensors[a_key], tensors[b_key]
            new_a, new_b, kept, _ = truncated_factors(a, b, threshold)
            # PEFT scaling is alpha/r. Concatenating the compressed and negative
            # original products therefore exactly represents Δ_k - Δ_original.
            output[a_key] = torch.cat((new_a, a), dim=0)
            output[b_key] = torch.cat((new_b, -b), dim=1)
            kept_ranks.append(kept)
        out_dir = args.output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(output, out_dir / "adapter_model.safetensors", metadata=metadata)
        out_config = dict(config)
        out_config["r"] = 2 * source_rank
        out_config["lora_alpha"] = 2 * source_alpha
        (out_dir / "adapter_config.json").write_text(json.dumps(out_config, indent=2) + "\n")
        summary[label] = {
            "threshold": threshold,
            "correction_rank": 2 * source_rank,
            "mean_retained_rank": sum(kept_ranks) / len(kept_ranks),
            "min_retained_rank": min(kept_ranks),
            "max_retained_rank": max(kept_ranks),
        }
        print(label, summary[label], flush=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
