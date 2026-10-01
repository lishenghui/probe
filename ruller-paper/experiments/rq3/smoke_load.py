#!/usr/bin/env python3
"""Load one RQ3 adapter, merge it, and run one deterministic generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["llm", "vlm"], required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--remamba-root", type=Path)
    args = parser.parse_args()

    if args.modality == "vlm":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        processor = AutoProcessor.from_pretrained(args.base)
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "Answer with OK."}]}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        print(processor.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True))
    else:
        if args.remamba_root is None:
            parser.error("--remamba-root is required for llm")
        sys.path.insert(0, str(args.remamba_root.resolve()))
        from src.model.ReMamba import ReMambaForCausalLM
        from src.model.configuration_ReMambahf import ReMambaConfig
        from transformers import AutoTokenizer, GenerationMixin

        class CompatibleReMambaForCausalLM(ReMambaForCausalLM, GenerationMixin):
            """Restore the GenerationMixin inherited by Transformers < 4.50."""

            @classmethod
            def _supports_default_dynamic_cache(cls):
                return False

        config_dir = args.remamba_root / "remambacfg" / "remamba"
        config = ReMambaConfig.from_pretrained(config_dir)
        config.ratio, config.stratio, config.compressp_ratio = 0.009, 0.0, 0.18
        model = CompatibleReMambaForCausalLM.from_pretrained(
            args.base, config=config, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
        inputs.pop("attention_mask", None)
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        print(tokenizer.decode(output[0, inputs.input_ids.shape[1] :]))


if __name__ == "__main__":
    main()
