#!/usr/bin/env python3
"""Merge one MBPP LoRA condition into Qwen2.5 for EvalPlus evaluation."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(args.base, revision=args.revision).save_pretrained(args.output)
    print(f"merged={args.output}")


if __name__ == "__main__":
    main()
