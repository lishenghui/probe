#!/usr/bin/env python3
"""Why three LoRA Land adapters score below chance, and whether a space fixes it.

CoLA, MNLI and MRPC were excluded from the pool because their uncompressed
adapters do not reproduce published behaviour -- all three score below the task's
chance rate, which means the harness, not the adapter, is broken. The prompt text
in TASKS matches the published model card verbatim and the label conventions
match the HF dataset, so the remaining candidate is the trailing separator: the
card ends its sample input with "Label: " and TASKS ends it with "Label:".

For Mistral's tokenizer that is not cosmetic. "Label:" followed by a generated
" 1" and "Label: " followed by "1" are different token sequences, and an adapter
trained on one can fail on the other.

This measures both, on the same rows, for the three failing tasks and one
working task as a control.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predibase_task_metrics import TASKS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tasks", nargs="+",
                    default=["glue_cola", "glue_mnli", "glue_mrpc", "glue_qnli"])
    ap.add_argument("--examples", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset

    def rows_for(spec, n):
        """Follow the same alt/revision fallbacks the evaluator uses.

        Several of these datasets are script-based and newer `datasets` refuses
        to run them; the recipe already carries a parquet mirror for each, and a
        probe that ignores it reports a dataset failure as a task failure.
        """
        last = None
        for ds_args, rev in [(spec["ds"], None)] + [((r,), rv) for r, rv in spec.get("alt", [])]:
            kw = {"split": f"{spec['split']}[:{n}]"}
            if rev:
                kw["revision"] = rev
            try:
                return load_dataset(*ds_args, **kw)
            except Exception as exc:
                last = f"{type(exc).__name__} {str(exc)[:70]}"
        raise SystemExit(f"no loadable source: {last}")
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to("cuda").eval()
    peft = None
    out = {}
    for task in args.tasks:
        spec = TASKS[task]
        rows = rows_for(spec, args.examples)
        golds = [spec["gold"](r) for r in rows]
        d = Path(snapshot_download(f"predibase/{task}"))
        if peft is None:
            peft = PeftModel.from_pretrained(model, str(d), adapter_name=task)
        else:
            peft.load_adapter(str(d), adapter_name=task)
        peft.set_adapter(task)
        res = {}
        for tag, suffix in (("as coded", ""), ("card separator", " ")):
            prompts = [spec["prompt"](r) + suffix for r in rows]
            hits, sample = [], []
            for i in range(0, len(prompts), args.batch):
                enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to("cuda")
                with torch.no_grad():
                    g = peft.generate(**enc, max_new_tokens=spec.get("new", 4),
                                      do_sample=False, pad_token_id=tok.pad_token_id)
                for j, s in enumerate(g[:, enc["input_ids"].shape[1]:]):
                    txt = tok.decode(s, skip_special_tokens=True).strip()
                    pred = txt.split()[0] if txt.split() else ""
                    gold = golds[i + j]
                    ok = pred == gold
                    if not ok:
                        # the cards answer some numeric tasks in float form ("3.0"
                        # for gold "3"), so a string comparison scores a correct
                        # answer as wrong
                        try:
                            ok = abs(float(pred) - float(gold)) < 1e-9
                        except (TypeError, ValueError):
                            ok = False
                    hits.append(float(ok))
                    if len(sample) < 4:
                        sample.append((repr(txt[:14]), golds[i + j]))
            res[tag] = dict(acc=sum(hits) / len(hits), sample=sample)
            print(f"{task:11s} {tag:16s} acc={res[tag]['acc']:.3f}  "
                  f"chance={spec.get('chance', float('nan')):.2f}  first: {sample[:3]}",
                  flush=True)
        peft.delete_adapter(task)
        torch.cuda.empty_cache()
        out[task] = res
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
