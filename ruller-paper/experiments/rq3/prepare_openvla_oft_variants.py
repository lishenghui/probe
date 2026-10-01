#!/usr/bin/env python3
"""Create E100/E99/E95/E90 OpenVLA-OFT checkpoint skeletons.

The source snapshot should contain the original ``lora_adapter`` plus the OFT
action head, proprio projector, and config/processor assets. Merged model shards
are deliberately neither required nor copied: every output is subsequently
merged from the same pristine OpenVLA base, including E100.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from compress_adapter import main as compress_main


MERGED_MODEL_NAMES = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
MERGED_MODEL_PREFIXES = ("model-", "pytorch_model-")


def is_merged_weight(path: Path) -> bool:
    return path.name in MERGED_MODEL_NAMES or path.name.startswith(MERGED_MODEL_PREFIXES)


def copy_common_assets(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == "lora_adapter" or is_merged_weight(item):
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.95, 0.90])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = args.source / "lora_adapter"
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise FileNotFoundError(f"Missing source LoRA under {adapter}")

    # Reuse the tested compressor through its CLI entry point.
    import sys
    saved_argv = sys.argv
    sys.argv = [
        "compress_adapter.py", "--weights", str(weights), "--config", str(config),
        "--output-root", str(args.output_root / "compressed"), "--thresholds",
        *[str(x) for x in args.thresholds],
    ]
    try:
        compress_main()
    finally:
        sys.argv = saved_argv

    labels = ["e100", *[f"e{round(x * 100):02d}" for x in args.thresholds]]
    for label in labels:
        destination = args.output_root / label
        copy_common_assets(args.source, destination)
        destination_adapter = destination / "lora_adapter"
        if destination_adapter.exists():
            shutil.rmtree(destination_adapter)
        if label == "e100":
            shutil.copytree(adapter, destination_adapter)
        else:
            shutil.copytree(args.output_root / "compressed" / label, destination_adapter)

    manifest = {
        "source": str(args.source.resolve()),
        "invariant": "all variants must be merged from the same pristine base",
        "variants": labels,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
