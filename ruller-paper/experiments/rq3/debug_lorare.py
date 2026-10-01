#!/usr/bin/env python3
"""Print raw generations for one LoraRetriever adapter, base vs adapted.

The eval set's `inputs` field ends with the FLAN options block and no answer cue,
so the first thing to rule out is whether the model needs a trailing newline to
switch from continuing the passage to answering it.
"""
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/nobackup/proj/disk/bloom/personal/shenghui/hf_cache/llama2-7b-hf"
rows = [r for r in load_dataset("Styxxxx/LoraRetriever_EvalSet", split="test")
        if r["task"].startswith("anli_r1")][:4]

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = tok.truncation_side = "left"
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
adir = snapshot_download("Styxxxx/llama2_7b_lora-anli_r1")
m = PeftModel.from_pretrained(model, adir, adapter_name="a")
m.set_adapter("a")

for suffix, tag in (("", "raw"), ("\n", "newline")):
    for disable in (True, False):
        ctx = m.disable_adapter() if disable else torch.no_grad()
        with ctx, torch.no_grad():
            enc = tok([r["inputs"] + suffix for r in rows], return_tensors="pt",
                      padding=True, truncation=True, max_length=1024).to("cuda")
            out = m.generate(**enc, max_new_tokens=24, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        print(f"\n### suffix={tag}  adapter={'OFF' if disable else 'ON'}")
        for r, seq in zip(rows, out[:, enc["input_ids"].shape[1]:]):
            print(f"   target={r['targets']!r}")
            print(f"   pred  ={tok.decode(seq, skip_special_tokens=True)!r}")
