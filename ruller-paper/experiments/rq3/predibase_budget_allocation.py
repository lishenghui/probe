#!/usr/bin/env python3
"""Uniform truncation vs strength-calibrated allocation, scored on task metrics.

Everything so far shows that adapter strength predicts how far compression moves
the model's output decisions.  It does not yet show that allocating by strength
buys anything a practitioner cares about, because the allocator has only ever
been evaluated against a divergence.  This script spends one fixed rank budget
two ways over the Lora Land pool and scores both on the tasks themselves.

The budget is exactly what uniform tau retains, summed over every module of
every adapter, so neither rule can win by keeping more directions.  Uniform
gives every adapter the same adapter-relative distortion; SCT chooses a per
adapter tau_i minimising the worst predicted damage

    log D_i = c + a log S_i + b log L_i(tau_i),

with (c, a, b) fitted on the *other* adapters -- leave-one-adapter-out, since
seven is too few to fold -- so no adapter is sized by a law its own divergence
helped fit.  The law is fitted against D_out (divergence at generated
decisions), not the prompt-prefix divergence it was originally fitted against.
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
from compress_adapter import truncated_factors  # noqa: E402
from predibase_task_metrics import TASKS, base_key, score, spectrum  # noqa: E402

L_MAX = 0.60      # the law is fitted over L in [0.07, 0.58]; do not extrapolate past it
L_MIN = 0.02


def load_cells(results: list[Path]):
    """(adapter, tau) -> S, L_W, d_out, taking d_out from the token probes."""
    cells, dirs = {}, [r.with_suffix("").with_name(r.stem + "_tokens") for r in results]
    for res in results:
        for r in json.loads(res.read_text()):
            for tau, v in r["variants"].items():
                f = next((d / f"{r['adapter']}-{tau}.npz" for d in dirs
                          if (d / f"{r['adapter']}-{tau}.npz").is_file()), None)
                if f is None or (r["adapter"], tau) in cells:
                    continue
                cells[(r["adapter"], tau)] = dict(
                    adapter=r["adapter"], S=r["S"], L_W=v["L_W"],
                    d_out=float(np.load(f)["per_token_js"].mean()))
    return list(cells.values())


def fit(cells):
    X = np.column_stack([np.log([c["S"] for c in cells]), np.log([c["L_W"] for c in cells]),
                         np.ones(len(cells))])
    y = np.log([c["d_out"] for c in cells])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta                                     # a, b, c


def curve(spectra):
    """k(tau) and L(tau) for one adapter under per-module energy truncation."""
    energies = {k: sv.square() for k, sv in spectra.items()}
    total = sum(float(e.sum()) for e in energies.values())

    def at(tau):
        k_tot, resid = 0, 0.0
        for e in energies.values():
            c = torch.cumsum(e, 0)
            k = int(torch.searchsorted(c, tau * c[-1]).item()) + 1
            k = min(k, e.numel())
            k_tot += k
            resid += float(e[k:].sum())
        return k_tot, math.sqrt(resid / total) if total > 0 else 0.0

    return at


def tabulate(adapters, n=2000):
    """(tau, k, L) for each adapter once; the allocator then only looks up."""
    taus = np.linspace(0.30, 0.999999, n)
    for ad in adapters.values():
        kl = [ad["at"](t) for t in taus]
        ad["tab"] = (taus, np.array([k for k, _ in kl]), np.array([L for _, L in kl]))


def allocate(adapters, budget, beta):
    """Per-adapter tau minimising the worst predicted damage at a fixed budget.

    Damage is monotone in L and L is monotone in tau, so the allocation for a
    given damage level is a lookup; the outer bisection matches the budget.
    """
    a, b, c = beta

    def alloc_for(level):
        out = {}
        for name, ad in adapters.items():
            taus, ks, Ls = ad["tab"]
            l_cap = min(math.exp((level - c - a * math.log(ad["S"])) / b), L_MAX)
            ok = np.flatnonzero(Ls <= l_cap)        # Ls descends as tau rises
            i = int(ok[0]) if ok.size else len(taus) - 1
            out[name] = dict(tau=float(taus[i]), k=int(ks[i]), L=float(Ls[i]))
        return out

    lo, hi = -40.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sum(v["k"] for v in alloc_for(mid).values()) > budget:
            lo = mid                                 # more damage allowed -> fewer directions
        else:
            hi = mid
    return alloc_for(hi)


def allocate_anchored(adapters, budget, anchors, b):
    """Minimax allocation after eliminating each adapter's intercept by an anchor."""
    def alloc_for(level):
        out = {}
        for name, ad in adapters.items():
            taus, ks, Ls = ad["tab"]
            ref = anchors[name]
            l_cap = min(ref["L_W"] * math.exp((level-math.log(ref["d_out"]))/b), L_MAX)
            ok = np.flatnonzero(Ls <= l_cap)
            i = int(ok[0]) if ok.size else len(taus)-1
            out[name] = dict(tau=float(taus[i]), k=int(ks[i]), L=float(Ls[i]))
        return out
    lo, hi = -40.0, 10.0
    for _ in range(60):
        mid=.5*(lo+hi)
        if sum(v["k"] for v in alloc_for(mid).values()) > budget: lo=mid
        else: hi=mid
    return alloc_for(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--match-tau", type=float, default=0.90)
    ap.add_argument("--anchor", type=Path, default=None,
                    help="disjoint-prompt anchor JSON; switches allocation to A-SCT")
    ap.add_argument("--fixed-b", type=float, default=None)
    ap.add_argument("--eval-shard", type=int, default=0)
    ap.add_argument("--eval-shards", type=int, default=1)
    ap.add_argument("--examples", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cells = load_cells(args.results)
    print(f"law fitted on {len(cells)} probed cells over "
          f"{len({c['adapter'] for c in cells})} adapters", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda").eval()
    norms = {n: float(p.detach().float().norm()) for n, p in model.named_parameters()}
    print(f"base loaded ({len(norms)} tensors)", flush=True)

    adapters = {}
    for task in args.tasks:
        spec = TASKS[task]
        rows = None
        for ds_args, rev in [(spec["ds"], None)] + [((r,), rv) for r, rv in spec.get("alt", [])]:
            try:
                kw = {"split": f"{spec['split']}[:{args.examples}]"}
                if rev:
                    kw["revision"] = rev
                rows = load_dataset(*ds_args, **kw)
                break
            except Exception:
                continue
        if rows is None:
            print(f"skip {task}: no loadable source", flush=True)
            continue
        local = Path(snapshot_download(f"predibase/{task}", local_dir=args.work / task))
        weights = load_file(local / "adapter_model.safetensors")
        cfg = json.loads((local / "adapter_config.json").read_text())
        rank, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
        scale = alpha / (math.sqrt(rank) if cfg.get("use_rslora") else rank)
        num = den = 0.0
        spectra = {}
        for a_key in sorted(k for k in weights if ".lora_A." in k):
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            w = norms.get(base_key(a_key))
            if w is None or b_key not in weights:
                continue
            sv = spectrum(weights[a_key], weights[b_key]) * scale
            spectra[a_key] = sv
            num += float(sv.square().sum())
            den += w * w
        adapters[task] = dict(
            S=math.sqrt(num / den), spectra=spectra, weights=weights, local=local,
            at=curve(spectra), nominal=sum(sv.numel() for sv in spectra.values()),
            prompts=[spec["prompt"](r) for r in rows], golds=[spec["gold"](r) for r in rows],
            spec=spec)
        print(f"  {task}: S={adapters[task]['S']:.4f} modules={len(spectra)} "
              f"nominal={adapters[task]['nominal']}", flush=True)

    tabulate(adapters)
    uniform = {n: dict(zip(("k", "L"), ad["at"](args.match_tau)), tau=args.match_tau)
               for n, ad in adapters.items()}
    budget = sum(v["k"] for v in uniform.values())
    print(f"\nbudget from uniform tau={args.match_tau}: {budget} of "
          f"{sum(ad['nominal'] for ad in adapters.values())} nominal directions", flush=True)

    sct = {}
    if args.anchor:
        if args.fixed_b is None: raise SystemExit("--anchor requires --fixed-b")
        raw=json.loads(args.anchor.read_text())
        anchors={r["adapter"]:r["variants"]["e95"] for r in raw}
        sct=allocate_anchored(adapters,budget,anchors,args.fixed_b)
        for name in adapters:
            sct[name].update(b=args.fixed_b, anchor_tau=.95)
            print(f"  {name}: anchored b={args.fixed_b:+.3f} -> tau={sct[name]['tau']:.4f} "
                  f"k={sct[name]['k']} (uniform {uniform[name]['k']}) "
                  f"L={sct[name]['L']:.3f} (uniform {uniform[name]['L']:.3f})",flush=True)
    else:
        # leave-one-adapter-out allocation: adapter i is sized by a law fitted without it
        for name in adapters:
            beta = fit([c for c in cells if c["adapter"] != name])
            got = allocate(adapters, budget, beta)
            sct[name] = got[name] | dict(a=float(beta[0]), b=float(beta[1]), c=float(beta[2]))
            print(f"  {name}: a={beta[0]:+.3f} b={beta[1]:+.3f} -> tau={sct[name]['tau']:.4f} "
                  f"k={sct[name]['k']} (uniform {uniform[name]['k']}) "
                  f"L={sct[name]['L']:.3f} (uniform {uniform[name]['L']:.3f})", flush=True)
    # Each adapter is sized by its own held-out law, so the seven allocations need
    # not sum to the budget.  Trim the overspend from the largest allocation --
    # that is the adapter SCT is protecting, so the correction cannot flatter it.
    spent = sum(v["k"] for v in sct.values())
    print(f"SCT spends {spent} vs budget {budget} ({spent - budget:+d}) before trimming",
          flush=True)
    while spent > budget:
        name = max(sct, key=lambda n: sct[n]["k"])
        taus, ks, Ls = adapters[name]["tab"]
        i = int(np.searchsorted(taus, sct[name]["tau"]))
        j = next((q for q in range(i - 1, -1, -1) if ks[q] < ks[i]), None)
        if j is None:
            break
        spent += int(ks[j]) - sct[name]["k"]
        sct[name].update(tau=float(taus[j]), k=int(ks[j]), L=float(Ls[j]))
    print(f"SCT spends {spent} vs budget {budget} ({spent - budget:+d}) after trimming",
          flush=True)

    peft = [None]

    def run(adapter_dir, name, ad):
        if peft[0] is None:
            peft[0] = PeftModel.from_pretrained(model, str(adapter_dir), adapter_name=name)
        else:
            peft[0].load_adapter(str(adapter_dir), adapter_name=name)
        m = peft[0]
        m.set_adapter(name)
        m.eval()
        hits = []
        for i in range(0, len(ad["prompts"]), args.batch):
            chunk = ad["prompts"][i:i + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length).to("cuda")
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=ad["spec"]["new"], do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for j, seq in enumerate(out[:, enc["input_ids"].shape[1]:]):
                hits.append(score(name.split("@")[0], tok.decode(seq, skip_special_tokens=True),
                                  ad["golds"][i + j]))
        m.delete_adapter(name)
        torch.cuda.empty_cache()
        return sum(hits) / max(len(hits), 1), hits

    results = []
    eval_names=sorted(adapters)[args.eval_shard::args.eval_shards]
    print(f"evaluation shard {args.eval_shard}/{args.eval_shards}: {eval_names}",flush=True)
    for name in eval_names:
        ad=adapters[name]
        m_orig, _ = run(ad["local"], f"{name}@orig", ad)
        rec = dict(adapter=name, S=ad["S"], metric_orig=m_orig, n=len(ad["prompts"]),
                   chance=ad["spec"]["chance"], rules={})
        print(f"\n{name:12s} S={ad['S']:.4f} orig={m_orig:.4f}", flush=True)
        for rule, alloc in (("uniform", uniform[name]), ("sct", sct[name])):
            out = dict(ad["weights"])
            for a_key in ad["spectra"]:
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                na, nb, _, _ = truncated_factors(ad["weights"][a_key], ad["weights"][b_key],
                                                 alloc["tau"])
                out[a_key], out[b_key] = na, nb
            dest = args.work / f"{name}-{rule}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text(
                (ad["local"] / "adapter_config.json").read_text())
            m_c, hits = run(dest, f"{name}@{rule}", ad)
            rel = (m_orig - m_c) / m_orig if m_orig > 0 else float("nan")
            rec["rules"][rule] = dict(alloc) | dict(metric=m_c, rel_drop=rel, per_example=hits)
            print(f"  {rule:8s} tau={alloc['tau']:.4f} k={alloc['k']:5d} L={alloc['L']:.3f} "
                  f"metric={m_c:.4f} rel_drop={rel:+.1%}", flush=True)
        results.append(rec)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            dict(budget=budget, match_tau=args.match_tau, adapters=results), indent=2) + "\n")

    # Relative drop is not comparable across these tasks: 0.955 -> 0.910 on SST-2
    # and 0.510 -> 0.485 on GSM8K are both -5%, but one is a fifth of the way to
    # chance and the other a twentieth.  Retention normalises by the headroom the
    # uncompressed adapter actually has over chance.
    def retention(rec, rule):
        head = rec["metric_orig"] - rec["chance"]
        if head <= 0:
            return float("nan")
        return (rec["rules"][rule]["metric"] - rec["chance"]) / head

    print(f"\n{'rule':10s} {'mean ret':>9} {'worst':>8} {'p10':>8} {'median':>8} "
          f"{'<0.9':>6} {'broken':>7} {'mean drop':>10} {'worst drop':>11}")
    for rule in ("uniform", "sct"):
        R = np.array([retention(r, rule) for r in results])
        d = np.array([r["rules"][rule]["rel_drop"] for r in results])
        lost = int((R < 0.9).sum())
        broken = int(sum(r["rules"][rule]["metric"] <= r["chance"] + 1e-9 for r in results))
        print(f"{rule:10s} {R.mean():9.3f} {R.min():8.3f} {np.percentile(R, 10):8.3f} "
              f"{np.median(R):8.3f} {lost:6d} {broken:7d} {d.mean():10.1%} {d.max():11.1%}")
    print(f"\n{'adapter':12s} {'S':>7} {'orig':>7} " +
          " ".join(f"{r + ' ' + c:>12}" for r in ("uni", "sct") for c in ("metric", "ret")))
    for r in results:
        print(f"{r['adapter']:12s} {r['S']:7.4f} {r['metric_orig']:7.4f} " +
              " ".join(f"{r['rules'][rule]['metric']:12.4f} {retention(r, rule):12.3f}"
                       for rule in ("uniform", "sct")))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
