#!/usr/bin/env python3
"""Pick the answer cue for the LoraRetriever pool before spending GPU on a sweep.

The eval set's `inputs` field ends with the FLAN options block and no separator.
Handed that string verbatim the adapter emits EOS immediately -- every prediction
is the empty string and the task metric is exactly 0.0000, below the un-adapted
model. The adapters were trained on `input + sep + target`, so the cue is part of
the harness, not a detail.

This is the check that the earlier Super-NaturalInstructions pool did not get: a
prompt-format mismatch does not crash, it silently reports that compression
destroyed an adapter that was never working.
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lorare_task_metrics import score  # noqa: E402

SEPS = {"raw": "", "nl": "\n", "nl2": "\n\n", "answer": "\nAnswer: ", "dash": "\n- "}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tasks", nargs="+",
                    default=["anli_r1", "wmt16_translate_de_en", "story_cloze"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    rows = load_dataset("Styxxxx/LoraRetriever_EvalSet", split="test")
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    picked = [t for t in sorted(by_task) if any(t.startswith(p) for p in args.tasks)]
    print("tasks:", picked, flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = tok.truncation_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to("cuda").eval()

    peft = None
    for task in picked:
        items = by_task[task][: args.n]
        metric = items[0]["metric"]
        short = task.rsplit("_", 1)[0] if task.split("_")[-1].endswith("templates") else task
        import re
        short = re.sub(r"_\d+templates$", "", task)
        adir = snapshot_download(f"Styxxxx/llama2_7b_lora-{short}")
        if peft is None:
            peft = PeftModel.from_pretrained(model, adir, adapter_name="a")
        else:
            peft.delete_adapter("a")
            peft.load_adapter(adir, adapter_name="a")
        peft.set_adapter("a")

        print(f"\n{short}  [{metric}]  n={len(items)}", flush=True)
        for name, sep in SEPS.items():
            line = []
            for disable in (True, False):
                hits = []
                ctx = peft.disable_adapter() if disable else torch.no_grad()
                with ctx:
                    for i in range(0, len(items), args.batch):
                        chunk = items[i:i + args.batch]
                        enc = tok([r["inputs"] + sep for r in chunk], return_tensors="pt",
                                  padding=True, truncation=True, max_length=1024).to("cuda")
                        with torch.no_grad():
                            out = peft.generate(**enc, max_new_tokens=args.new,
                                                do_sample=False,
                                                pad_token_id=tok.pad_token_id)
                        for r, seq in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
                            hits.append(score(metric,
                                              tok.decode(seq, skip_special_tokens=True).strip(),
                                              r["targets"]))
                line.append(sum(hits) / max(len(hits), 1))
            base_m, ad_m = line
            print(f"  sep={name:7s} base={base_m:.4f}  adapter={ad_m:.4f}  "
                  f"gain={ad_m - base_m:+.4f}", flush=True)


if __name__ == "__main__":
    main()
