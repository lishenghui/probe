#!/usr/bin/env python3
"""Strength-calibrated truncation (SCT): allocate one rank budget over a pool.

Given adapters i with spectra fixed, choose retained ranks k_i minimising the
predicted functional damage under the fitted scaling law

    log D_i = c + a log S_i + b log L_i(k_i),      L_i(k) = sqrt(1 - E_i(k)),

subject to sum_i k_i <= B.  Because L_i(k) is decreasing in k and the objective
is separable, a greedy pass on the marginal decrease per direction is exact for
the sum objective; we also expose the minimax variant.

Equal predicted damage across the pool means S_i^{a/b} L_i = const, so the
special case a = b recovers "equalise the model-relative perturbation
P_i = S_i L_i" -- but using the *achieved* L_i rather than the nominal
sqrt(1 - tau), which is what the earlier form got wrong.

Exponents come from folds of the pool that exclude the adapter being allocated,
so no adapter is sized using a law its own divergence helped fit.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402
from sweep_cts_compression import js_divergence  # noqa: E402


def ols(X, Y):
    A = [[1.0] + list(x) for x in X]
    m = len(A[0])
    M = [[sum(A[i][p] * A[i][q] for i in range(len(A))) for q in range(m)] for p in range(m)]
    v = [sum(A[i][p] * Y[i] for i in range(len(A))) for p in range(m)]
    for c in range(m):
        pv = max(range(c, m), key=lambda r: abs(M[r][c]))
        M[c], M[pv] = M[pv], M[c]
        v[c], v[pv] = v[pv], v[c]
        d = M[c][c]
        M[c] = [x / d for x in M[c]]
        v[c] /= d
        for r in range(m):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [p - f * q for p, q in zip(M[r], M[c])]
                v[r] -= f * v[c]
    return v


def curves(spectra):
    """Per-module cumulative energy, and the pool-level L(k) machinery."""
    out = []
    for sv in spectra:
        e = sv.square()
        tot = float(e.sum())
        out.append((e, tot))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--lw", type=Path, required=True, help="cts_scaling_intervention.json")
    ap.add_argument("--strengths", type=Path, required=True,
                    help="cts_strength_aggregations.json, for the global-norm S")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--match-tau", type=float, default=0.90)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--objective", choices=("minimax", "sum"), default="minimax",
                    help="minimise the worst predicted damage, or their sum.  The sum "
                         "objective starves weak adapters: their marginal gain scales as "
                         "S^a, so a greedy pass never spends on them.")
    ap.add_argument("--l-max", type=float, default=0.35,
                    help="never truncate an adapter past this adapter-relative residual; "
                         "the scaling law is fitted for L in [0.07, 0.30] and must not be "
                         "extrapolated to L -> 1.")
    ap.add_argument("--fold", type=int, required=True, help="which fold this shard evaluates")
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--truncation-side", choices=("left", "right"), default="left",
                    help="SuperNI prompts end with the task instance, so the default "
                         "right truncation removes it: at max_length=320 that collapsed "
                         "20 of 30 pool adapters' 48 prompts to one shared preamble")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32",
                    help="bf16 logits floor the weak adapters' divergence, which flattens "
                         "the fitted exponents and mis-calibrates the allocator.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lw = {r["adapter"]: r for r in json.loads(args.lw.read_text())["L_W"]}
    agg = {r["adapter"]: r for r in json.loads(args.strengths.read_text())}
    entries = sorted((r for r in json.loads(args.sweep.read_text())
                      if r["adapter"] in lw and r["adapter"] in agg),
                     key=lambda r: agg[r["adapter"]]["S_global"])
    for rank, e in enumerate(entries):
        e["fold"] = rank % args.folds
        e["S"] = agg[e["adapter"]]["S_global"]

    # Fit the law on the other folds only.
    train = [e for e in entries if e["fold"] != args.fold]
    X, Y = [], []
    for e in train:
        for t in ("e99", "e95", "e90"):
            X.append((math.log(e["S"]), math.log(lw[e["adapter"]][f"L_{t}"])))
            Y.append(math.log(e["variants"][t]["d_js_mean"]))
    c, a, b = ols(X, Y)
    print(f"fold {args.fold}: fitted on {len(train)} held-in adapters -> "
          f"a={a:.3f} b={b:.3f} a/b={a/b:.3f}", flush=True)

    members = [e for e in entries if e["fold"] == args.fold]
    for e in members:
        e["local"] = Path(snapshot_download(
            f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{e['task']}",
            local_dir=args.work / f"task{e['task']:04d}"))
        w = load_file(e["local"] / "adapter_model.safetensors")
        e["weights"] = w
        e["energy"] = []
        for a_key in sorted(k for k in w if ".lora_A." in k):
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            a32, b32 = w[a_key].float(), w[b_key].float()
            qb, rb = torch.linalg.qr(b32, mode="reduced")
            qa, ra = torch.linalg.qr(a32.T, mode="reduced")
            sv = torch.linalg.svdvals(rb @ ra.T)
            e["energy"].append((a_key, sv.square()))

    def L_of(e, ranks):
        kept = sum(float(en[:k].sum()) for (_, en), k in zip(e["energy"], ranks))
        tot = sum(float(en.sum()) for _, en in e["energy"])
        return math.sqrt(max(tot - kept, 0.0) / tot) if tot > 0 else 0.0

    def damage(e, ranks):
        L = L_of(e, ranks)
        return math.exp(c + a * math.log(e["S"]) + b * math.log(max(L, 1e-8)))

    # Budget: what uniform truncation at match_tau retains on this fold.
    budget = 0
    baseline = {}
    for e in members:
        rk = []
        for _, en in e["energy"]:
            tot = float(en.sum())
            cum = torch.cumsum(en, 0) / tot if tot > 0 else en
            rk.append(int(torch.searchsorted(cum, args.match_tau).item()) + 1)
        baseline[e["adapter"]] = rk
        budget += sum(rk)

    # Allocation.  Start every adapter at the rank that meets --l-max, which keeps
    # the surrogate inside the range it was fitted on, then spend what is left.
    ranks, spent = {}, 0
    for e in members:
        rk = []
        for _, en in e["energy"]:
            tot = float(en.sum())
            cum = torch.cumsum(en, 0) / tot if tot > 0 else en
            rk.append(int(torch.searchsorted(cum, 1.0 - args.l_max ** 2).item()) + 1)
        ranks[e["adapter"]] = rk
        spent += sum(rk)
    if spent > budget:
        raise SystemExit(f"floor at L<={args.l_max} needs {spent} > budget {budget}")

    idx = {e["adapter"]: e for e in members}
    if args.objective == "minimax":
        # Give the next direction to whoever is currently worst off.  The module
        # inside that adapter is the one whose next direction removes most energy.
        while spent < budget:
            key = max(ranks, key=lambda k: damage(idx[k], ranks[k]))
            e = idx[key]
            best, gain = None, 0.0
            for m, (_, en) in enumerate(e["energy"]):
                k = ranks[key][m]
                if k < len(en) and float(en[k]) > gain:
                    best, gain = m, float(en[k])
            if best is None:
                break
            ranks[key][best] += 1
            spent += 1
    else:
        heap = []
        for e in members:
            cur = damage(e, ranks[e["adapter"]])
            for m in range(len(e["energy"])):
                if ranks[e["adapter"]][m] < len(e["energy"][m][1]):
                    t = list(ranks[e["adapter"]]); t[m] += 1
                    heapq.heappush(heap, (-(cur - damage(e, t)), e["adapter"], m))
        while spent < budget and heap:
            g, key, m = heapq.heappop(heap)
            e = idx[key]
            if ranks[key][m] >= len(e["energy"][m][1]):
                continue
            cur = damage(e, ranks[key])
            t = list(ranks[key]); t[m] += 1
            act = cur - damage(e, t)
            if -g > act + 1e-18:
                heapq.heappush(heap, (-act, key, m)); continue
            ranks[key][m] += 1; spent += 1
            if ranks[key][m] < len(e["energy"][m][1]):
                c2 = damage(e, ranks[key]); t2 = list(ranks[key]); t2[m] += 1
                heapq.heappush(heap, (-(c2 - damage(e, t2)), key, m))
    Ls = sorted(L_of(e, ranks[e["adapter"]]) for e in members)
    print(f"fold {args.fold}: objective={args.objective} budget {budget} allocated {spent}; "
          f"L range {Ls[0]:.3f}-{Ls[-1]:.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.truncation_side = args.truncation_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dt).to("cuda").eval()
    peft_model, results = None, []

    for e in members:
        w = e["weights"]
        out = dict(w)
        for (a_key, _), k in zip(e["energy"], ranks[e["adapter"]]):
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            new_a, new_b = truncated_factors(w[a_key], w[b_key], 1.0)[:2]
            # truncated_factors picks k by energy; redo the projection at fixed k
            a32, b32 = w[a_key].float(), w[b_key].float()
            qb, rb = torch.linalg.qr(b32, mode="reduced")
            qa, ra = torch.linalg.qr(a32.T, mode="reduced")
            uc, sv, vhc = torch.linalg.svd(rb @ ra.T, full_matrices=False)
            root = sv[:k].sqrt()
            na = torch.zeros_like(a32)
            nb = torch.zeros_like(b32)
            nb[:, :k] = (qb @ uc[:, :k]) * root.unsqueeze(0)
            na[:k, :] = root.unsqueeze(1) * (vhc[:k, :] @ qa.T)
            out[a_key] = na.to(w[a_key].dtype)
            out[b_key] = nb.to(w[b_key].dtype)
        dest = args.work / f"task{e['task']:04d}-sct"
        dest.mkdir(parents=True, exist_ok=True)
        save_file(out, dest / "adapter_model.safetensors")
        (dest / "adapter_config.json").write_text((e["local"] / "adapter_config.json").read_text())

        rows = load_dataset(e["dataset"], split="train")
        field = next(x for x in ("input", "text", "prompt") if x in rows.column_names)
        texts = [r for r in rows[field][: args.prompts * 3] if r and r.strip()][: args.prompts]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length)
        parts = [{k: v[i:i + args.chunk].to("cuda") for k, v in enc.items()}
                 for i in range(0, len(texts), args.chunk)]

        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(e["local"]), adapter_name="orig")
        else:
            peft_model.load_adapter(str(e["local"]), adapter_name="orig")
        peft_model.load_adapter(str(dest), adapter_name="sct")
        peft_model.eval()

        js_all, flips, seen = [], 0, 0
        for part in parts:
            mask = part["attention_mask"].bool()
            with torch.no_grad():
                peft_model.set_adapter("orig")
                r = peft_model(**part)
                peft_model.set_adapter("sct")
                v = peft_model(**part)
            js_all.append(js_divergence(r.logits, v.logits)[mask].cpu())
            flips += int((r.logits.argmax(-1)[mask] != v.logits.argmax(-1)[mask]).sum())
            seen += int(mask.sum())
            del r, v
        js = torch.cat(js_all)
        results.append({
            "adapter": e["adapter"], "task": e["task"], "fold": args.fold,
            "S": e["S"], "a": a, "b": b,
            "L_sct": L_of(e, ranks[e["adapter"]]),
            "L_uniform": L_of(e, baseline[e["adapter"]]),
            "rank_sct": sum(ranks[e["adapter"]]), "rank_uniform": sum(baseline[e["adapter"]]),
            "objective": args.objective,
            "d_js_mean": float(js.mean()), "argmax_flip": flips / max(seen, 1),
            "uniform_d_js_mean": e["variants"][f"e{round(args.match_tau*100):02d}"]["d_js_mean"],
        })
        r0 = results[-1]
        print(f"task{e['task']:04d} S={e['S']:.4f} L {r0['L_uniform']:.3f}->{r0['L_sct']:.3f} "
              f"rank {r0['rank_uniform']}->{r0['rank_sct']}  JS {r0['uniform_d_js_mean']:.5f}"
              f"->{r0['d_js_mean']:.5f}", flush=True)
        for name in ("orig", "sct"):
            peft_model.delete_adapter(name)
        torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
