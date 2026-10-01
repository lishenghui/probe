#!/usr/bin/env python3
"""Measure one LoRA Land adapter's activation sensitivity and dense E2E curve.

Calibration prompts are unlabeled.  Sensitivity is measured once on the full
adapter; every compressed point reuses that fixed vector.  D_JS is measured on
the same prompts between the full adapter and the sensitivity-allocated one.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402
from predibase_task_metrics import TASKS  # noqa: E402
from sweep_cts_compression import js_divergence  # noqa: E402
from two_level_allocation import allocate_ranks, allocate_squared_ranks  # noqa: E402


def adapter_module_name(key: str) -> str:
    name = key.rsplit(".lora_A.", 1)[0]
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pool", choices=("land", "cts", "lorare"), default="land")
    ap.add_argument("--cue", default="\n\n",
                    help="LoraRetriever only: the separator its eval set omits. Without "
                         "it the adapter emits EOS immediately and every divergence is "
                         "measured against an empty continuation.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument("--example-start", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--layer-risk", choices=("norm", "squared", "spectral", "functional"), default="norm",
                    help="norm uses marginal greedy on S^gamma L; squared uses the "
                         "exact global top-K solution for S^gamma normalized energy; "
                         "spectral reproduces original A-SCT raw singular-energy allocation")
    ap.add_argument("--layer-costs", type=Path,
                    help="isolated-layer probe JSON required by --layer-risk functional")
    ap.add_argument("--divergence", choices=("prompt", "output"), default="prompt",
                    help="measure JS on prompt tokens or on full-adapter generated-token "
                         "decisions under a shared teacher-forced prefix")
    ap.add_argument("--record-decision-risk", action="store_true",
                    help="also record token- and sequence-level argmax disagreement "
                         "against the full adapter for every measured pattern")
    ap.add_argument("--budget-step", type=int, default=8)
    ap.add_argument("--max-budget", type=int, default=256)
    ap.add_argument("--min-rank", type=int, choices=(0, 1), default=1,
                    help="minimum retained rank per LoRA module; zero allows the "
                         "allocator to disable an unneeded LoRA branch")
    ap.add_argument("--new", type=int, default=None,
                    help="generation length; defaults to the LoRA Land task setting or 32 for CTS")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.pool == "land":
        if args.task not in TASKS:
            raise ValueError(f"unknown LoRA Land task: {args.task}")
        spec = TASKS[args.task]
        end = args.example_start + args.examples
        rows = None
        attempts = [(spec["ds"], None)] + [((repo,), rev) for repo, rev in spec.get("alt", [])]
        for ds_args, revision in attempts:
            try:
                kw = {"split": f"{spec['split']}[{args.example_start}:{end}]"}
                if revision:
                    kw["revision"] = revision
                rows = load_dataset(*ds_args, **kw)
                break
            except Exception as exc:
                last_error = exc
        if rows is None:
            raise RuntimeError(f"could not load {args.task}: {last_error}")
        prompts = [spec["prompt"](row) for row in rows]
        adapter_repo = f"predibase/{args.task}"
        generation_length = args.new or spec["new"]
    elif args.pool == "lorare":
        import re as _re
        rows = load_dataset("Styxxxx/LoraRetriever_EvalSet", split="test")
        items = [r for r in rows if r["task"] == args.task]
        if not items:
            raise RuntimeError(f"{args.task}: not in the eval set")
        window = items[args.example_start:args.example_start + args.examples]
        if len(window) < args.examples:
            raise RuntimeError(f"{args.task}: only {len(window)} calibration prompts")
        prompts = [r["inputs"] + args.cue for r in window]
        short = _re.sub(r"_\d+templates$", "", args.task)
        adapter_repo = f"Styxxxx/llama2_7b_lora-{short}"
        generation_length = args.new or 48
    else:
        from huggingface_hub import HfApi
        if not args.task.startswith("task"):
            raise ValueError("CTS adapter must be named taskNNNN")
        task_number = int(args.task.replace("task", ""))
        datasets = {}
        for info in HfApi().list_datasets(author="Lots-of-LoRAs"):
            slug = info.id.split("/")[-1]
            if slug.startswith("task"):
                digits = "".join(c for c in slug[4:] if c.isdigit())
                if digits:
                    datasets.setdefault(int(digits), info.id)
        if task_number not in datasets:
            raise RuntimeError(f"no dataset for {args.task}")
        available = load_dataset(datasets[task_number])
        split = "train" if "train" in available else next(iter(available))
        rows = available[split]
        field = next((name for name in ("input", "text", "prompt")
                      if name in rows.column_names), None)
        if field is None:
            raise RuntimeError(f"{args.task}: no prompt field")
        candidates = [text for text in rows[field] if text and text.strip()]
        prompts = candidates[args.example_start:args.example_start + args.examples]
        if len(prompts) < args.examples:
            raise RuntimeError(f"{args.task}: only {len(prompts)} calibration prompts")
        adapter_repo = ("Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task"
                        f"{task_number}")
        generation_length = args.new or 32

    args.work.mkdir(parents=True, exist_ok=True)
    local = Path(snapshot_download(adapter_repo,
                                   local_dir=args.work / args.task.replace("/", "_")))
    weights = load_file(local / "adapter_model.safetensors")
    config = json.loads((local / "adapter_config.json").read_text())
    rank = int(config["r"])
    scale = float(config["lora_alpha"]) / (math.sqrt(rank) if config.get("use_rslora") else rank)
    a_keys = sorted(k for k in weights if ".lora_A." in k)
    sigma = []
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        a, b = weights[a_key].float(), weights[b_key].float()
        qb, rb = torch.linalg.qr(b, mode="reduced")
        qa, ra = torch.linalg.qr(a.T, mode="reduced")
        sigma.append((torch.linalg.svdvals(rb @ ra.T) * scale).tolist())

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda").eval()
    model = PeftModel.from_pretrained(base, str(local), adapter_name="full").eval()

    # Match PEFT modules to the same sorted tensor order used for rank patterns.
    peft_modules = {name: module for name, module in model.named_modules()
                    if hasattr(module, "lora_A") and "full" in module.lora_A}
    ordered_modules = []
    for key in a_keys:
        wanted = adapter_module_name(key)
        matches = [(name, module) for name, module in peft_modules.items()
                   if name == wanted or name.endswith("." + wanted)]
        if len(matches) != 1:
            raise RuntimeError(f"{key}: expected one PEFT module, got {[m[0] for m in matches]}")
        ordered_modules.append(matches[0])

    sums = torch.zeros(len(a_keys), dtype=torch.float64)
    counts = torch.zeros(len(a_keys), dtype=torch.int64)
    active_mask = [None]
    handles = []
    for module_index, (_, module) in enumerate(ordered_modules):
        def measure(mod, inputs, output, idx=module_index):
            x = inputs[0]
            # PEFT's full adapter contribution, including its configured dropout/scale.
            # LoraLayer.forward casts to the adapter dtype before applying A/B;
            # this hook must mirror that because the base runs in bf16 while
            # PEFT keeps these adapter weights in fp32 by default.
            adapter_dtype = mod.lora_A["full"].weight.dtype
            lora_input = mod.lora_dropout["full"](x).to(adapter_dtype)
            delta = mod.lora_B["full"](mod.lora_A["full"](lora_input))
            delta = delta * mod.scaling["full"]
            y = output[0] if isinstance(output, tuple) else output
            trunk = y - delta.to(y.dtype)
            ratio = delta.float().norm(dim=-1) / trunk.float().norm(dim=-1).clamp_min(1e-12)
            mask = active_mask[0]
            if mask is not None and ratio.ndim == 2 and mask.shape == ratio.shape:
                ratio = ratio[mask]
            sums[idx] += ratio.double().sum().cpu()
            counts[idx] += ratio.numel()
        handles.append(module.register_forward_hook(measure))

    model.set_adapter("full")
    for start in range(0, len(prompts), args.batch):
        enc = tok(prompts[start:start + args.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_length).to("cuda")
        active_mask[0] = enc["attention_mask"].bool()
        with torch.no_grad():
            model(**enc)
    active_mask[0] = None
    for handle in handles:
        handle.remove()
    sensitivity = (sums / counts.clamp_min(1)).tolist()
    if any(not math.isfinite(x) for x in sensitivity):
        raise RuntimeError("non-finite sensitivity")
    print(f"{args.task}: calibrated {len(sensitivity)} modules; "
          f"S range {min(sensitivity):.3e}--{max(sensitivity):.3e}", flush=True)

    floor, ceiling = len(a_keys) * args.min_rank, sum(len(s) for s in sigma)
    max_budget = min(args.max_budget, ceiling)
    budgets = list(range(floor, max_budget + 1, args.budget_step))
    if budgets[-1] != max_budget:
        budgets.append(max_budget)
    functional_costs = None
    if args.layer_risk == "functional":
        if args.layer_costs is None:
            raise ValueError("--layer-risk functional requires --layer-costs")
        probe = json.loads(args.layer_costs.read_text())
        if probe.get("adapter") != args.task:
            raise ValueError(f"layer-cost adapter mismatch: {probe.get('adapter')} != {args.task}")
        if int(probe.get("modules", -1)) != len(sigma) or int(probe.get("r", -1)) != rank:
            raise ValueError("layer-cost module/rank shape mismatch")
        functional_costs = [[0.0] * (rank + 1) for _ in sigma]
        measured = [set() for _ in sigma]
        for row in probe["single_layer"]:
            module, measured_rank = int(row["module"]), int(row["k"])
            functional_costs[module][measured_rank] = float(row["d_js"])
            measured[module].add(measured_rank)
        if args.min_rank == 0 and any(0 not in ranks for ranks in measured):
            raise ValueError("functional min-rank 0 requires an isolated k=0 cost "
                             "for every module")
        for row in functional_costs:
            row[rank] = 0.0
        from two_level_allocation import allocate_functional_dp
        patterns = {k: allocate_functional_dp(
            functional_costs, k, min_rank=args.min_rank) for k in budgets}
    elif args.layer_risk == "squared":
        allocator = allocate_squared_ranks
    elif args.layer_risk == "spectral":
        from two_level_allocation import allocate_global_spectral_ranks
        allocator = allocate_global_spectral_ranks
    else:
        allocator = allocate_ranks
    if args.layer_risk != "functional":
        patterns = {k: allocator(sigma, sensitivity, k, args.gamma,
                                 min_rank=args.min_rank) for k in budgets}

    generated = None
    if args.divergence == "output":
        generated = []
        model.set_adapter("full")
        for start in range(0, len(prompts), args.batch):
            enc = tok(prompts[start:start + args.batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_length).to("cuda")
            with torch.no_grad():
                sequences = model.generate(
                    **enc, max_new_tokens=generation_length, do_sample=False,
                    pad_token_id=tok.pad_token_id)
            continuation = sequences[:, enc["input_ids"].shape[1]:]
            for seq in continuation:
                ids = seq.tolist()
                if tok.eos_token_id in ids:
                    ids = ids[:ids.index(tok.eos_token_id) + 1]
                generated.append(ids)
        print(f"{args.task}: generated {sum(map(len, generated))} calibration tokens", flush=True)

    def divergence(adapter_name: str) -> tuple[float, float, float]:
        total = seen = token_flips = sequence_flips = sequences_seen = 0
        for start in range(0, len(prompts), args.batch):
            chunk = prompts[start:start + args.batch]
            if args.divergence == "prompt":
                enc = tok(chunk, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length).to("cuda")
                positions = enc["attention_mask"].bool()
            else:
                assert generated is not None
                prompt_ids = [tok(p, truncation=True, max_length=args.max_length)["input_ids"]
                              for p in chunk]
                gens = generated[start:start + len(chunk)]
                full = [p + g for p, g in zip(prompt_ids, gens)]
                width = max(map(len, full))
                input_ids = torch.full((len(full), width), tok.pad_token_id, dtype=torch.long)
                attention = torch.zeros((len(full), width), dtype=torch.long)
                positions = torch.zeros((len(full), width), dtype=torch.bool)
                for row, (ids, gen) in enumerate(zip(full, gens)):
                    offset = width - len(ids)
                    input_ids[row, offset:] = torch.tensor(ids)
                    attention[row, offset:] = 1
                    # Token y_t is predicted at the preceding position.  Include
                    # EOS if generated, since emitting EOS is itself a decision.
                    first = offset + len(ids) - len(gen) - 1
                    positions[row, first:first + len(gen)] = True
                enc = {"input_ids": input_ids.to("cuda"),
                       "attention_mask": attention.to("cuda")}
                positions = positions.to("cuda")
            with torch.no_grad():
                model.set_adapter("full")
                ref = model(**enc).logits
                model.set_adapter(adapter_name)
                var = model(**enc).logits
            js = js_divergence(ref, var)[positions]
            total += float(js.sum())
            seen += int(positions.sum())
            if args.record_decision_risk:
                changed = (ref.argmax(-1) != var.argmax(-1)) & positions
                token_flips += int(changed.sum())
                sequence_flips += int(changed.any(-1).sum())
                sequences_seen += int(changed.shape[0])
            del ref, var
        return (total / max(seen, 1), token_flips / max(seen, 1),
                sequence_flips / max(sequences_seen, 1))

    curve = []
    for k in budgets:
        module_ranks, risk = patterns[k]
        variant_dir = args.work / f"{args.task}-twolevel-{k}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        out = dict(weights)
        kept = residual = 0.0
        for index, a_key in enumerate(a_keys):
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            na, nb, actual, _ = truncated_factors(
                weights[a_key], weights[b_key], 0.0, fixed_rank=module_ranks[index])
            out[a_key], out[b_key] = na, nb
            e = torch.tensor(sigma[index]).square()
            kept += float(e[:actual].sum())
            residual += float(e[actual:].sum())
        save_file(out, variant_dir / "adapter_model.safetensors")
        shutil.copy2(local / "adapter_config.json", variant_dir / "adapter_config.json")
        adapter_name = f"k{k}"
        model.load_adapter(str(variant_dir), adapter_name=adapter_name)
        d_js, token_flip, sequence_flip = divergence(adapter_name)
        model.delete_adapter(adapter_name)
        shutil.rmtree(variant_dir)
        row = {"k": k, "d_js": d_js, "weighted_risk": risk,
               "L_W": math.sqrt(residual / (kept + residual)),
               "module_ranks": module_ranks}
        if args.record_decision_risk:
            row.update(token_flip=token_flip, sequence_flip=sequence_flip)
        curve.append(row)
        extra = (f" token_flip={token_flip:.3%} seq_flip={sequence_flip:.3%}"
                 if args.record_decision_risk else "")
        print(f"  K={k:3d} D_JS={d_js:.6e} risk={risk:.4e}{extra}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "adapter": args.task, "pool": args.pool, "gamma": args.gamma,
            "examples": len(prompts),
            "example_start": args.example_start, "budget_step": args.budget_step,
            "min_rank": args.min_rank,
            "divergence": args.divergence, "layer_risk": args.layer_risk,
            "layer_costs": str(args.layer_costs) if args.layer_costs else None,
            "module_keys": a_keys, "sensitivity": sensitivity, "sigma": sigma,
            "curve": curve,
        }, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
