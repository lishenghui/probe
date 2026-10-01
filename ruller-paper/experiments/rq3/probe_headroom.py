#!/usr/bin/env python3
"""Does the adapter beat the un-adapted model by enough to measure retention?

Retained utility divides by m_full - m_base, so an adapter whose headroom is near
zero contributes a noise-amplified ratio no matter how carefully the sweep is run.
Generation tasks are the risk: a base model already scores respectable token-F1 on
summarisation, so a new recipe can be perfectly correct and still be unusable.

This runs base and full only, which is a twelfth of the cost of the sweep it
decides whether to launch.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predibase_task_metrics import TASKS, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--examples", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-headroom", type=float, default=0.05)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to("cuda").eval()
    peft, out = None, {}
    for task in args.tasks:
        spec = TASKS[task]
        rows = None
        for ds, rev in [(spec["ds"], None)] + [((r,), rv) for r, rv in spec.get("alt", [])]:
            kw = {"split": f"{spec['split']}[:{args.examples}]"}
            if rev:
                kw["revision"] = rev
            try:
                rows = load_dataset(*ds, **kw); break
            except Exception:
                pass
        if rows is None:
            print(f"{task}: no loadable source", flush=True); continue
        prompts = [spec["prompt"](r) for r in rows]
        golds = [spec["gold"](r) for r in rows]
        d = snapshot_download(f"predibase/{task}")
        if peft is None:
            peft = PeftModel.from_pretrained(model, d, adapter_name=task)
        else:
            peft.load_adapter(d, adapter_name=task)
        peft.set_adapter(task)

        def run(disable):
            hits = []
            ctx = peft.disable_adapter() if disable else torch.no_grad()
            with ctx:
                for i in range(0, len(prompts), args.batch):
                    enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True,
                              truncation=True, max_length=1024).to("cuda")
                    with torch.no_grad():
                        g = peft.generate(**enc, max_new_tokens=spec.get("new", 16),
                                          do_sample=False, pad_token_id=tok.pad_token_id)
                    for j, sq in enumerate(g[:, enc["input_ids"].shape[1]:]):
                        hits.append(score(task, tok.decode(sq, skip_special_tokens=True),
                                          golds[i + j]))
            return sum(hits) / len(hits)

        b, f = run(True), run(False)
        ok = (f - b) >= args.min_headroom
        out[task] = dict(base=b, full=f, headroom=f - b, usable=ok)
        print(f"{task:22s} base={b:.4f} full={f:.4f} headroom={f-b:+.4f}  "
              f"{'USABLE' if ok else 'below threshold'}", flush=True)
        peft.delete_adapter(task)
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{sum(v['usable'] for v in out.values())}/{len(out)} usable")


if __name__ == "__main__":
    main()
