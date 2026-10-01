#!/usr/bin/env python3
"""How much of the primary-pool divergence depends on the prompt truncation?

`sweep_cts_compression.py` tokenises with `truncation=True, max_length=320` and
the tokenizer's default right truncation.  SuperNI prompts are longer than that
-- median 406 tokens, max 1603 -- and they end with the task instance:

    Definition: <instruction> ... Positive Example 1 ... Negative Example 2 ...
    Now complete the following example -
    Input: <the actual instance>          <- cut

so for 21 of the 30 pool adapters every prompt is truncated and all 48 collapse
to the same shared preamble.  Verified directly: for those adapters the
per-token margins are bit-identical across prompts.

This measures the same adapters under both settings in one process -- same base
model, same weights, same thresholds, only the tokenisation differs -- so the
comparison is not confounded by anything else.  The question is not whether the
old numbers were computed correctly; it is whether the conclusion drawn from them
survives being measured on prompts that still contain the task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402
from sweep_cts_compression import js_divergence  # noqa: E402

SETTINGS = {"old": dict(max_length=320, side="right"),
            "fixed": dict(max_length=2048, side="left")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--strengths", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.95, 0.90])
    ap.add_argument("--adapters", type=int, default=0,
                    help="evenly spaced over the pool's strength order; 0 takes all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1,
                    help="strided, so each shard still spans the strength range and "
                         "a partial result is not a partial range")
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--split", default="train",
                    help="train, to reproduce the published sweep's input source")
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi()
    by_task = {}
    for info in api.list_datasets(author="Lots-of-LoRAs"):
        slug = info.id.split("/")[-1]
        if slug.startswith("task"):
            d = "".join(c for c in slug[4:] if c.isdigit())
            if d:
                by_task.setdefault(int(d), info.id)

    entries = sorted(json.loads(args.strengths.read_text()), key=lambda r: r["mean"])
    if args.adapters:
        idx = np.linspace(0, len(entries) - 1, args.adapters).round().astype(int)
        entries = [entries[i] for i in sorted(set(idx.tolist()))]
    entries = entries[args.shard::args.shards]
    print(f"shard {args.shard}/{args.shards}: {len(entries)} adapters, S from "
          f"{entries[0]['mean']:.4f} to {entries[-1]['mean']:.4f}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to("cuda").eval()
    print("base loaded", flush=True)

    peft_model, results = None, []
    for entry in entries:
        task = int(entry["adapter"].replace("task", ""))
        if task not in by_task:
            print(f"skip task{task:04d}: no dataset", flush=True)
            continue
        try:
            local = Path(snapshot_download(
                f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{task}",
                local_dir=args.work / f"task{task:04d}"))
            rows = load_dataset(by_task[task], split=args.split)
        except Exception as exc:
            print(f"skip task{task:04d}: {type(exc).__name__}", flush=True)
            continue
        field = next((c for c in ("input", "text", "prompt") if c in rows.column_names), None)
        texts = [r for r in rows[field][: args.prompts * 3] if r and r.strip()][: args.prompts]
        if len(texts) < 8:
            print(f"skip task{task:04d}: only {len(texts)} prompts", flush=True)
            continue

        weights = load_file(local / "adapter_model.safetensors")
        a_keys = sorted(k for k in weights if ".lora_A." in k)
        variants = {}
        for threshold in args.thresholds:
            label = f"e{round(threshold * 100):02d}"
            out = dict(weights)
            for a_key in a_keys:
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                na, nb, _, _ = truncated_factors(weights[a_key], weights[b_key], threshold)
                out[a_key], out[b_key] = na, nb
            dest = args.work / f"task{task:04d}-{label}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text(
                (local / "adapter_config.json").read_text())
            variants[label] = dest

        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(local), adapter_name="orig")
        else:
            peft_model.load_adapter(str(local), adapter_name="orig")
        m = peft_model
        for label, d in variants.items():
            m.load_adapter(str(d), adapter_name=label)
        m.eval()

        record = {"adapter": entry["adapter"], "S": entry["mean"], "task": task,
                  "prompts": len(texts), "settings": {}}
        for sname, cfg in SETTINGS.items():
            tok.truncation_side = cfg["side"]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=cfg["max_length"])
            n_trunc = sum(len(tok(t)["input_ids"]) > cfg["max_length"] for t in texts)
            # if every prompt collapses to the same token sequence, the 48 draws
            # are one draw; report it rather than let it hide in the average
            uniq = len({tuple(r[msk].tolist()) for r, msk in
                        zip(enc["input_ids"], enc["attention_mask"].bool())})
            per = {"truncated": n_trunc, "unique_prompts": uniq,
                   "max_length": cfg["max_length"], "side": cfg["side"], "variants": {}}
            for label in variants:
                tot, seen, flips = 0.0, 0, 0
                for i in range(0, len(texts), args.chunk):
                    part = {k: v[i:i + args.chunk].to("cuda") for k, v in enc.items()}
                    msk = part["attention_mask"].bool()
                    with torch.no_grad():
                        m.set_adapter("orig")
                        a = m(**part).logits
                        m.set_adapter(label)
                        b = m(**part).logits
                    tot += float(js_divergence(a, b)[msk].sum())
                    flips += int((a.argmax(-1)[msk] != b.argmax(-1)[msk]).sum())
                    seen += int(msk.sum())
                    del a, b
                torch.cuda.empty_cache()
                per["variants"][label] = {"d_js_mean": tot / max(seen, 1),
                                          "flip": flips / max(seen, 1), "positions": seen}
            record["settings"][sname] = per
            print(f"task{task:04d} S={entry['mean']:.4f} [{sname:5s}] "
                  f"trunc={n_trunc}/{len(texts)} unique={uniq} " +
                  " ".join(f"{k}={v['d_js_mean']:.3e}" for k, v in per["variants"].items()),
                  flush=True)
        for label in list(variants) + ["orig"]:
            m.delete_adapter(label)
        torch.cuda.empty_cache()
        results.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    # the comparison that decides whether the conclusion moves; with shards this
    # is done by the merge step instead, on the union
    print("\n--- fit under each setting (this shard only) ---")
    lw = {}
    lwf = Path("artifacts/rq3/results/cts_scaling_intervention.json")
    if lwf.is_file():
        lw = {r["adapter"]: r for r in json.loads(lwf.read_text())["L_W"]}
    for sname in SETTINGS:
        pts = [(r["S"], lw[r["adapter"]][f"L_{t}"], v["d_js_mean"])
               for r in results if r["adapter"] in lw
               for t, v in r["settings"][sname]["variants"].items()]
        if len(pts) < 6:
            print(f"  {sname}: too few points"); continue
        X = np.column_stack([np.log([p[0] for p in pts]), np.log([p[1] for p in pts]),
                             np.ones(len(pts))])
        y = np.log([p[2] for p in pts])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        b1, *_ = np.linalg.lstsq(X[:, 1:], y, rcond=None)
        r2l = 1 - ((y - X[:, 1:] @ b1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {sname:5s} n={len(pts):3d}  a={beta[0]:+.2f} b={beta[1]:+.2f} "
              f"R2={r2:.3f}  (L_W alone {r2l:.3f}, S adds {r2 - r2l:+.3f})")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
