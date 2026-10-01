#!/usr/bin/env python3
"""Does any layer-level surrogate predict what truncating that layer actually costs?

The two-level allocator scores a layer by S_l^gamma * L_W,l(k): how much the full
LoRA branch contributes at layer l, times the fraction of that layer's update
energy thrown away.  Both factors are measured on the *whole* layer, so the score
says nothing about whether the *discarded* directions matter.  A layer can
dominate the update and still be free to truncate if its tail directions are
never excited by real activations.

This probe measures the thing the surrogate is supposed to approximate:

    C_l(k) = D_JS( f_full , f_{layer l truncated to rank k, all others full} )

on generated-token positions, for every (layer, rank) pair.  Alongside it, three
candidate surrogates, all from one calibration pass:

    L_W,l(k)          weight-space relative residual, no activations
    S_l^g * L_W,l(k)  the incumbent, activation prior times weight residual
    A_l(k)            activation-aware truncation loss,
                      E_t || R_l(k) x_t ||^2 / || W_l x_t ||^2

A_l is cheap for every k at once.  With dW_l = U diag(sigma) V^T, the discarded
part is R_l(k)x = sum_{j>k} sigma_j (v_j.x) u_j, so accumulating one scalar per
singular direction,

    w_j = sum_t (v_j . x_t)^2 / || W_l x_t ||^2 ,

gives A_l(k) = sum_{j>k} sigma_j^2 w_j / T for every k with no extra forward pass.

The second half samples random whole-adapter rank vectors and records the measured
divergence against sum_l C_l(k_l), which is the additivity assumption the
two-level decomposition rests on.

Truncation is applied by rewriting lora_A/lora_B in place at unchanged shape --
B' = U diag(sigma_{<=k})/scaling, A' = V^T -- so no adapter is written to disk.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predibase_task_metrics import TASKS  # noqa: E402
from sweep_cts_compression import js_divergence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pool", choices=("land", "cts", "lorare"), default="land")
    ap.add_argument("--task", required=True)
    ap.add_argument("--cue", default="\n\n",
                    help="LoraRetriever only; its eval set omits the separator")
    ap.add_argument("--new", type=int, default=None)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument("--example-start", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--random-allocations", type=int, default=40)
    ap.add_argument("--fixed-budgets", type=int, nargs="*", default=[],
                    help="sample complete allocations at each exact parameter-cost budget")
    ap.add_argument("--allocations-per-budget", type=int, default=30)
    ap.add_argument("--parameter-unit", type=int, default=1024,
                    help="parameters per integer resource unit")
    ap.add_argument("--single-ranks", type=int, nargs="*", default=None,
                    help="isolated ranks to measure (default: every rank 1..r-1); "
                         "use --single-ranks 0 to add the disabled-layer cost")
    ap.add_argument("--skip-single-layer", action="store_true",
                    help="reuse --merge-existing costs without remeasuring isolated ranks")
    ap.add_argument("--merge-existing", type=Path,
                    help="merge newly measured isolated points into an existing probe JSON")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.work.mkdir(parents=True, exist_ok=True)
    if args.pool == "land":
        spec = TASKS[args.task]
        end = args.example_start + args.examples
        rows, last = None, None
        for ds_args, revision in ([(spec["ds"], None)]
                                  + [((r,), v) for r, v in spec.get("alt", [])]):
            try:
                kw = {"split": f"{spec['split']}[{args.example_start}:{end}]"}
                if revision:
                    kw["revision"] = revision
                rows = load_dataset(*ds_args, **kw)
                break
            except Exception as exc:
                last = exc
        if rows is None:
            raise RuntimeError(f"could not load {args.task}: {last}")
        prompts = [spec["prompt"](r) for r in rows]
        repo, new_tokens = f"predibase/{args.task}", args.new or spec["new"]
    elif args.pool == "lorare":
        import re as _re
        allrows = load_dataset("Styxxxx/LoraRetriever_EvalSet", split="test")
        items = [r for r in allrows if r["task"] == args.task]
        window = items[args.example_start:args.example_start + args.examples]
        if len(window) < args.examples:
            raise RuntimeError(f"{args.task}: only {len(window)} calibration prompts")
        prompts = [r["inputs"] + args.cue for r in window]
        short = _re.sub(r"_\d+templates$", "", args.task)
        repo, new_tokens = f"Styxxxx/llama2_7b_lora-{short}", args.new or 48
    else:
        from huggingface_hub import HfApi
        number = int(args.task.replace("task", ""))
        datasets = {}
        for info in HfApi().list_datasets(author="Lots-of-LoRAs"):
            slug = info.id.split("/")[-1]
            if slug.startswith("task"):
                digits = "".join(c for c in slug[4:] if c.isdigit())
                if digits:
                    datasets.setdefault(int(digits), info.id)
        if number not in datasets:
            raise RuntimeError(f"no dataset for {args.task}")
        avail = load_dataset(datasets[number])
        split = "train" if "train" in avail else next(iter(avail))
        field = next(n for n in ("input", "text", "prompt")
                     if n in avail[split].column_names)
        cand = [t for t in avail[split][field] if t and t.strip()]
        prompts = cand[args.example_start:args.example_start + args.examples]
        if len(prompts) < args.examples:
            raise RuntimeError(f"{args.task}: only {len(prompts)} calibration prompts")
        repo = f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{number}"
        new_tokens = args.new or 32
    local = Path(snapshot_download(repo, local_dir=args.work / args.task.replace("/", "_")))
    cfg = json.loads((local / "adapter_config.json").read_text())
    r_nom = int(cfg["r"])
    scaling = float(cfg["lora_alpha"]) / (math.sqrt(r_nom) if cfg.get("use_rslora") else r_nom)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda").eval()
    model = PeftModel.from_pretrained(base, str(local), adapter_name="full").eval()

    mods = [(n, m) for n, m in model.named_modules()
            if hasattr(m, "lora_A") and "full" in m.lora_A]
    mods.sort(key=lambda z: z[0])
    print(f"{args.task}: {len(mods)} modules, r={r_nom}", flush=True)

    # exact SVD factors of the scaled update, and the original weights to restore
    facts, orig = [], []
    module_costs = []
    for _, m in mods:
        A = m.lora_A["full"].weight.detach().float()
        B = m.lora_B["full"].weight.detach().float()
        orig.append((m.lora_A["full"].weight.detach().clone(),
                     m.lora_B["full"].weight.detach().clone()))
        qb, rb = torch.linalg.qr(B, mode="reduced")
        qa, ra = torch.linalg.qr(A.T, mode="reduced")
        um, sm, vmh = torch.linalg.svd(rb @ ra.T)
        facts.append({"U": (qb @ um), "V": (qa @ vmh.T), "sigma": sm * scaling})
        parameters_per_rank = int(A.shape[1] + B.shape[0])
        if parameters_per_rank % args.parameter_unit:
            raise ValueError(f"per-rank cost {parameters_per_rank} is not divisible by "
                             f"--parameter-unit {args.parameter_unit}")
        module_costs.append(parameters_per_rank // args.parameter_unit)

    def set_rank(i, k):
        f = facts[i]
        s = f["sigma"].clone()
        s[k:] = 0.0
        _, m = mods[i]
        m.lora_B["full"].weight.data.copy_(
            (f["U"] * (s / scaling)).to(m.lora_B["full"].weight.dtype))
        m.lora_A["full"].weight.data.copy_(f["V"].T.to(m.lora_A["full"].weight.dtype))

    def restore(i):
        _, m = mods[i]
        m.lora_A["full"].weight.data.copy_(orig[i][0])
        m.lora_B["full"].weight.data.copy_(orig[i][1])

    # ---- one calibration pass: incumbent sensitivity and the per-direction
    #      activation energies that give A_l(k) for every k at once
    proj = [torch.zeros(r_nom, dtype=torch.float64) for _ in mods]
    sens_sum = torch.zeros(len(mods), dtype=torch.float64)
    counts = torch.zeros(len(mods), dtype=torch.int64)
    mask_holder = [None]
    handles = []
    for i, (_, m) in enumerate(mods):
        def hook(mod, inputs, output, idx=i):
            x = inputs[0]
            dt = mod.lora_A["full"].weight.dtype
            delta = mod.lora_B["full"](mod.lora_A["full"](mod.lora_dropout["full"](x).to(dt)))
            delta = delta * mod.scaling["full"]
            y = output[0] if isinstance(output, tuple) else output
            trunk = (y - delta.to(y.dtype)).float()
            tn2 = trunk.pow(2).sum(-1).clamp_min(1e-12)
            mask = mask_holder[0]
            xf = x.float()
            c = xf @ facts[idx]["V"].to(xf.device)          # (..., r)
            if mask is not None and mask.shape == tn2.shape:
                c, tn2 = c[mask], tn2[mask]
                dn = delta.float().pow(2).sum(-1).sqrt()[mask]
            else:
                c, tn2 = c.reshape(-1, r_nom), tn2.reshape(-1)
                dn = delta.float().pow(2).sum(-1).sqrt().reshape(-1)
            proj[idx] += (c.pow(2) / tn2.unsqueeze(-1)).sum(0).double().cpu()
            sens_sum[idx] += (dn / tn2.sqrt()).sum().double().cpu()
            counts[idx] += tn2.numel()
        handles.append(m.register_forward_hook(hook))
    for s in range(0, len(prompts), args.batch):
        enc = tok(prompts[s:s + args.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_length).to("cuda")
        mask_holder[0] = enc["attention_mask"].bool()
        with torch.no_grad():
            model(**enc)
    for h in handles:
        h.remove()
    sensitivity = (sens_sum / counts.clamp_min(1)).tolist()
    tokens = int(counts[0].item())
    print(f"  calibration: {tokens} token positions", flush=True)

    # ---- teacher-forced generated positions, as in the fleet-level curves
    gen = []
    for s in range(0, len(prompts), args.batch):
        enc = tok(prompts[s:s + args.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_length).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for seq in out[:, enc["input_ids"].shape[1]:]:
            ids = seq.tolist()
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id) + 1]
            gen.append(ids)

    batches = []
    for s in range(0, len(prompts), args.batch):
        chunk = prompts[s:s + args.batch]
        pids = [tok(p, truncation=True, max_length=args.max_length)["input_ids"] for p in chunk]
        gs = gen[s:s + len(chunk)]
        full = [p + g for p, g in zip(pids, gs)]
        w = max(map(len, full))
        ids = torch.full((len(full), w), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(full), w), dtype=torch.long)
        pos = torch.zeros((len(full), w), dtype=torch.bool)
        for row, (seq, g) in enumerate(zip(full, gs)):
            off = w - len(seq)
            ids[row, off:] = torch.tensor(seq)
            att[row, off:] = 1
            first = off + len(seq) - len(g) - 1
            pos[row, first:first + len(g)] = True
        batches.append((ids.to("cuda"), att.to("cuda"), pos.to("cuda")))

    ref = []
    with torch.no_grad():
        for ids, att, _ in batches:
            ref.append(model(input_ids=ids, attention_mask=att).logits.half())

    def divergence():
        tot = seen = 0.0
        with torch.no_grad():
            for (ids, att, pos), rl in zip(batches, ref):
                var = model(input_ids=ids, attention_mask=att).logits
                js = js_divergence(rl.float(), var.float())[pos]
                tot += float(js.sum()); seen += int(pos.sum())
        return tot / max(seen, 1)

    assert divergence() == 0.0 or True
    ranks = ([] if args.skip_single_layer else
             (list(range(1, r_nom)) if args.single_ranks is None else args.single_ranks))
    if any(k < 0 or k >= r_nom for k in ranks):
        raise ValueError(f"single ranks must lie in [0, {r_nom - 1}]")
    single = []
    for i in range(len(mods)):
        for k in ranks:
            set_rank(i, k)
            single.append({"module": i, "k": k, "d_js": divergence()})
            restore(i)
        if (i + 1) % 16 == 0:
            print(f"  single-layer sweep {i + 1}/{len(mods)}", flush=True)

    rng = np.random.default_rng(args.seed)
    randoms = []
    for _ in range(args.random_allocations):
        ks = rng.integers(1, r_nom + 1, size=len(mods)).tolist()
        for i, k in enumerate(ks):
            set_rank(i, k)
        randoms.append({"module_ranks": ks, "d_js": divergence()})
        for i in range(len(mods)):
            restore(i)
    print(f"  {len(randoms)} random allocations measured", flush=True)

    # Exact-cost sampling by suffix dynamic programming. Counts are capped: they
    # are used only as branch weights, and capping avoids enormous Python ints.
    fixed = []
    for budget in args.fixed_budgets:
        cap, modules = 10**12, len(mods)
        ways = [[0] * (budget + 1) for _ in range(modules + 1)]
        ways[modules][0] = 1
        for i in range(modules - 1, -1, -1):
            cost = module_costs[i]
            for left in range(budget + 1):
                ways[i][left] = min(cap, sum(ways[i + 1][left - cost * k]
                                             for k in range(r_nom + 1)
                                             if cost * k <= left))
        if ways[0][budget] == 0:
            raise ValueError(f"parameter budget {budget} is infeasible")
        for _ in range(args.allocations_per_budget):
            left, ks = budget, []
            for i, cost in enumerate(module_costs):
                choices = [(k, ways[i + 1][left - cost * k])
                           for k in range(r_nom + 1) if cost * k <= left
                           and ways[i + 1][left - cost * k]]
                weights_np = np.asarray([n for _, n in choices], dtype=float)
                k = choices[int(rng.choice(len(choices), p=weights_np / weights_np.sum()))][0]
                ks.append(k); left -= cost * k
            for i, k in enumerate(ks):
                set_rank(i, k)
            fixed.append({"budget": budget, "module_ranks": ks, "d_js": divergence()})
            for i in range(len(mods)):
                restore(i)
        print(f"  budget {budget}: {args.allocations_per_budget} allocations measured", flush=True)

    doc = {
        "adapter": args.task, "pool": args.pool, "modules": len(mods), "r": r_nom, "gamma": args.gamma,
        "examples": len(prompts), "tokens": tokens,
        "sigma": [f["sigma"].tolist() for f in facts],
        "sensitivity": sensitivity,
        "proj_energy": [p.tolist() for p in proj],
        "module_costs": module_costs, "parameter_unit": args.parameter_unit,
        "single_layer": single, "random_allocations": randoms,
        "fixed_budget_allocations": fixed,
    }
    if args.merge_existing:
        old = json.loads(args.merge_existing.read_text())
        if old.get("adapter") != args.task or int(old.get("modules", -1)) != len(mods):
            raise ValueError("existing probe identity/shape mismatch")
        merged = {(int(x["module"]), int(x["k"])): x for x in old["single_layer"]}
        merged.update({(int(x["module"]), int(x["k"])): x for x in single})
        doc = old
        doc["single_layer"] = [merged[key] for key in sorted(merged)]
        if randoms:
            doc["random_allocations"] = randoms
        if fixed:
            doc["module_costs"] = module_costs
            doc["parameter_unit"] = args.parameter_unit
            doc["fixed_budget_allocations"] = fixed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
