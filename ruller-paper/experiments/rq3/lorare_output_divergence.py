#!/usr/bin/env python3
"""Prompt- and output-position divergence for the LoRARetriever adapter pool.

For every adapter, the uncompressed adapter greedily generates a continuation.
The original and each spectral truncation are then compared (1) over the supplied
prompt and (2) over that same original continuation.  Thus D_out is a
shared-prefix measurement, not divergence accumulated along different rollouts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402
from sweep_cts_compression import js_divergence  # noqa: E402

EVAL_SET = "Styxxxx/LoraRetriever_EvalSet"
ADAPTER = "Styxxxx/llama2_7b_lora-{task}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--strengths", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[.99, .95, .90])
    ap.add_argument("--prompts", type=int, default=50)
    ap.add_argument("--start", type=int, default=0,
                    help="offset within each task; use a nonzero offset to keep "
                         "unlabeled calibration prompts disjoint from task evaluation")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--cue", default="\n\n")
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    strength = {r["task"]: r for r in json.loads(args.strengths.read_text())}
    data = load_dataset(EVAL_SET, split="test")
    by_task = defaultdict(list)
    for row in data:
        by_task[row["task"]].append(row)
    tasks = sorted(by_task)[args.shard::args.shards]
    print(f"{len(by_task)} tasks; shard {args.shard}/{args.shards} takes {len(tasks)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Model execution and JS accumulation are both float32. This avoids the
    # bfloat16 numerical floor observed for the weakest adapters in this pool.
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).cuda().eval()
    peft_model, results = None, []
    print("base loaded in fp32", flush=True)

    for task in tasks:
        items = by_task[task][args.start:args.start + args.prompts]
        if not items:
            print(f"skip {task}: no rows in [{args.start}, "
                  f"{args.start + args.prompts}) out of {len(by_task[task])}", flush=True)
            continue
        prompts = [r["inputs"] + args.cue for r in items]
        short = re.sub(r"_\d+templates$", "", task)
        if short not in strength:
            print(f"skip {task}: no global strength", flush=True)
            continue
        try:
            local = Path(snapshot_download(ADAPTER.format(task=short), local_dir=args.work / short))
        except Exception as exc:
            print(f"skip {task}: {type(exc).__name__}: {exc}"[:180], flush=True)
            continue

        weights = load_file(local / "adapter_model.safetensors")
        a_keys = sorted(k for k in weights if ".lora_A" in k)
        variants = {}
        for threshold in args.thresholds:
            label = f"e{round(threshold * 100):02d}"
            out, kept, drop, k_tot, n_tot = dict(weights), 0., 0., 0, 0
            for a_key in a_keys:
                b_key = a_key.replace(".lora_A", ".lora_B")
                na, nb, k, _ = truncated_factors(weights[a_key], weights[b_key], threshold)
                out[a_key], out[b_key] = na, nb
                a32, b32 = weights[a_key].float(), weights[b_key].float()
                _, rb = torch.linalg.qr(b32, mode="reduced")
                _, ra = torch.linalg.qr(a32.T, mode="reduced")
                energy = torch.linalg.svdvals(rb @ ra.T).square()
                kept += float(energy[:k].sum()); drop += float(energy[k:].sum())
                k_tot += k; n_tot += energy.numel()
            dest = args.work / f"{short}-{label}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text((local / "adapter_config.json").read_text())
            variants[label] = dict(path=dest, L_W=(drop / (kept + drop)) ** .5,
                                   rank_frac=k_tot / n_tot)

        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(local), adapter_name="orig")
        else:
            peft_model.load_adapter(str(local), adapter_name="orig")
        m = peft_model.eval()
        m.set_adapter("orig")
        generations = []
        for i in range(0, len(prompts), args.gen_batch):
            enc = tok(prompts[i:i + args.gen_batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_length).to("cuda")
            with torch.no_grad():
                got = m.generate(**enc, max_new_tokens=args.new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for seq in got[:, enc["input_ids"].shape[1]:]:
                ids = seq.tolist()
                if tok.eos_token_id in ids:
                    ids = ids[:ids.index(tok.eos_token_id) + 1]
                generations.append(ids)
        print(f"{short}: {len(prompts)} prompts, {sum(map(len, generations))} output tokens", flush=True)

        record = dict(task=task, short=short, S=strength[short]["S"], prompts=len(prompts),
                      prompt_start=args.start,
                      gen_tokens=sum(map(len, generations)), cue=args.cue, variants={})
        for label, variant in variants.items():
            m.load_adapter(str(variant["path"]), adapter_name=label)
            p_tot = o_tot = 0.; p_seen = o_seen = 0
            for i in range(0, len(prompts), args.chunk):
                chunk, gens = prompts[i:i + args.chunk], generations[i:i + args.chunk]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_length)
                cuda = {k: v.cuda() for k, v in enc.items()}
                with torch.no_grad():
                    m.set_adapter("orig"); la = m(**cuda).logits
                    m.set_adapter(label); lb = m(**cuda).logits
                mask = cuda["attention_mask"].bool()
                p_tot += float(js_divergence(la, lb)[mask].sum()); p_seen += int(mask.sum())
                del la, lb

                keep = enc["attention_mask"].bool()
                full = [enc["input_ids"][j][keep[j]].tolist() + list(gens[j])
                        for j in range(len(chunk))]
                width = max(map(len, full))
                inp = torch.full((len(full), width), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros_like(inp)
                for j, seq in enumerate(full):
                    inp[j, width-len(seq):] = torch.tensor(seq); att[j, width-len(seq):] = 1
                inp, att = inp.cuda(), att.cuda()
                with torch.no_grad():
                    m.set_adapter("orig"); la = m(input_ids=inp, attention_mask=att).logits
                    m.set_adapter(label); lb = m(input_ids=inp, attention_mask=att).logits
                for j, gen in enumerate(gens):
                    if not gen:
                        continue
                    start = width - len(gen)
                    pos = torch.arange(start - 1, start - 1 + len(gen), device="cuda")
                    o_tot += float(js_divergence(la[j, pos], lb[j, pos]).sum())
                    o_seen += len(gen)
                del la, lb
            m.delete_adapter(label); torch.cuda.empty_cache()
            record["variants"][label] = dict(
                L_W=variant["L_W"], rank_frac=variant["rank_frac"],
                d_prompt=p_tot / p_seen, d_out=o_tot / o_seen,
                prompt_positions=p_seen, out_positions=o_seen)
            v = record["variants"][label]
            print(f"  {label}: Dp={v['d_prompt']:.3e} Do={v['d_out']:.3e} "
                  f"L={v['L_W']:.3f}", flush=True)
        m.delete_adapter("orig"); torch.cuda.empty_cache()
        results.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}: {len(results)} tasks", flush=True)


if __name__ == "__main__":
    main()
