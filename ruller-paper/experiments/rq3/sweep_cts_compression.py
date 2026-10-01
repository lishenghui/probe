#!/usr/bin/env python3
"""Does the VLA failure mode reappear inside Compress-then-Serve's own pool?

CTS reports mean utility preserved over a heterogeneous adapter collection, and
that collection spans an order of magnitude in adapter strength
S = ||alpha * B A||_F / ||W||_F (0.015 to 0.167 across the Lots-of-LoRAs release,
with the OpenVLA adapter that energy truncation breaks sitting at its 82nd
percentile).  If the VLA result is about strength rather than about robotics,
CTS's own strong adapters should degrade like the VLA under the same truncation.

Every adapter is probed on its *own* task data, so what varies across the sweep
is the adapter, not the input distribution.  The base model is loaded once and
adapters are swapped in and out of it, which is what makes 33 adapters x 3
thresholds affordable in a single job.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = torch.log_softmax(p_logits.float(), dim=-1)
    q = torch.log_softmax(q_logits.float(), dim=-1)
    m = torch.logaddexp(p, q) - math.log(2.0)
    return 0.5 * ((p.exp() * (p - m)).sum(-1) + (q.exp() * (q - m)).sum(-1))


def base_name(a_key: str) -> str:
    name = a_key.replace(".lora_A.weight", ".weight").replace(".lora_A.default.weight", ".weight")
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--strengths", type=Path, required=True,
                    help="cts_adapter_strength.json; sets which adapters to sweep")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.95, 0.90])
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--ranks", type=int, nargs="*", default=[],
                    help="also emit fixed-rank variants keeping exactly k directions "
                         "per module, the other compression rule in common use")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--truncation-side", choices=("left", "right"), default="left",
                    help="SuperNI prompts end with the task instance, so the default "
                         "right truncation removes it: at max_length=320 that collapsed "
                         "20 of 30 pool adapters' 48 prompts to one shared preamble")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                    help="bf16 logits put a floor under small divergences; on weak-adapter "
                         "populations that floor flattens the fitted exponents.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi()
    # Adapter repos are named by task number; the probe data lives in a dataset
    # repo whose name carries the task's slug, so the number is the join key.
    ds_by_task = {}
    for info in api.list_datasets(author="Lots-of-LoRAs"):
        slug = info.id.split("/")[-1]
        if slug.startswith("task"):
            digits = slug[4:].split("_")[0]
            if digits.isdigit():
                ds_by_task.setdefault(int(digits), info.id)

    entries = sorted(json.loads(args.strengths.read_text()), key=lambda r: r["mean"])
    entries = entries[args.shard::args.shards]
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.truncation_side = args.truncation_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dt).to("cuda").eval()
    print(f"base loaded; sweeping {len(entries)} adapters", flush=True)

    peft_model = None
    results = []
    for entry in entries:
        task = int(entry["adapter"].replace("task", ""))
        repo = f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{task}"
        dataset_id = ds_by_task.get(task)
        if dataset_id is None:
            print(f"skip task{task:04d}: no dataset", flush=True)
            continue
        try:
            local = Path(snapshot_download(repo, local_dir=args.work / f"task{task:04d}"))
            rows = load_dataset(dataset_id, split="train")
        except Exception as exc:                      # a handful of repos are missing
            print(f"skip task{task:04d}: {type(exc).__name__} {exc}"[:160], flush=True)
            continue

        field = next((c for c in ("input", "text", "prompt") if c in rows.column_names), None)
        texts = [r for r in rows[field][: args.prompts * 3] if r and r.strip()][: args.prompts]
        if len(texts) < 8:
            print(f"skip task{task:04d}: only {len(texts)} prompts", flush=True)
            continue
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length)
        parts = [{k: v[i:i + args.chunk].to("cuda") for k, v in enc.items()}
                 for i in range(0, len(texts), args.chunk)]

        # Energy-truncate in memory; every variant keeps the nominal rank so PEFT
        # can load them all against the same config.
        weights = load_file(local / "adapter_model.safetensors")
        a_keys = sorted(k for k in weights if ".lora_A." in k)
        variants = {"orig": local}
        rules = ([(f"e{round(t * 100):02d}", t, None) for t in args.thresholds]
                 + [(f"k{k:02d}", None, k) for k in args.ranks])
        for label, threshold, fixed in rules:
            out = dict(weights)
            kept_total = nominal_total = 0
            energy_kept = energy_total = 0.0
            for a_key in a_keys:
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                new_a, new_b, kept, retained = truncated_factors(
                    weights[a_key], weights[b_key], threshold or 0.0, fixed_rank=fixed)
                energy_kept += retained
                energy_total += 1.0
                out[a_key], out[b_key] = new_a, new_b
                kept_total += kept
                nominal_total += weights[a_key].shape[0]
            dest = args.work / f"task{task:04d}-{label}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text(
                (local / "adapter_config.json").read_text())
            variants[label] = dest
            entry[f"rank_frac_{label}"] = kept_total / max(nominal_total, 1)
            entry[f"retained_{label}"] = energy_kept / max(energy_total, 1)

        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(variants["orig"]),
                                                   adapter_name="orig")
        else:
            peft_model.load_adapter(str(variants["orig"]), adapter_name="orig")
        for label in variants:
            if label != "orig":
                peft_model.load_adapter(str(variants[label]), adapter_name=label)
        peft_model.eval()

        record = {"adapter": entry["adapter"], "S": entry["mean"], "task": task,
                  "dataset": dataset_id, "prompts": len(texts), "variants": {}}
        for label in variants:
            if label == "orig":
                continue
            js_all, hid_all, flips, seen = [], [], 0, 0
            for part in parts:
                mask = part["attention_mask"].bool()
                with torch.no_grad():
                    peft_model.set_adapter("orig")
                    ref = peft_model(**part, output_hidden_states=True)
                    peft_model.set_adapter(label)
                    var = peft_model(**part, output_hidden_states=True)
                js = js_divergence(ref.logits, var.logits)[mask]
                js_all.append(js.cpu())
                rh, vh = ref.hidden_states[-1].float()[mask], var.hidden_states[-1].float()[mask]
                hid_all.append(((vh - rh).norm(dim=-1) / rh.norm(dim=-1).clamp_min(1e-8)).cpu())
                flips += int((ref.logits.argmax(-1)[mask] != var.logits.argmax(-1)[mask]).sum())
                seen += int(mask.sum())
                del ref, var
            js = torch.cat(js_all)
            hid = torch.cat(hid_all)
            record["variants"][label] = {
                "d_js_mean": float(js.mean()), "d_js_p90": float(js.quantile(0.9)),
                "d_hidden": float(hid.mean()), "argmax_flip": flips / max(seen, 1),
                "rank_frac": entry.get(f"rank_frac_{label}"),
                "retained_energy": entry.get(f"retained_{label}"),
            }
        results.append(record)
        line = "  ".join(f"{k}: JS={v['d_js_mean']:.4f} flip={v['argmax_flip']:.1%} "
                         f"rank={v['rank_frac']:.2f}" for k, v in record["variants"].items())
        print(f"task{task:04d} S={entry['mean']:.4f}  {line}", flush=True)

        for label in variants:
            peft_model.delete_adapter(label)
        torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\nwrote {args.output} ({len(results)} adapters)")


if __name__ == "__main__":
    main()
