#!/usr/bin/env python3
"""Validate the strength-adjusted threshold at a matched parameter budget.

Uniform truncation gives every adapter the same *adapter*-relative distortion.
Eq. (3) instead gives every adapter the same *model*-relative one:

    P_i = S_i sqrt(1 - tau_i) <= eps   <=>   tau_i >= 1 - (eps / S_i)^2,

so eps is the single knob.  We bisect eps so that the total retained rank over
the pool equals what uniform tau = 0.90 retains, which makes the comparison a
reallocation of one fixed budget across adapters rather than a change of budget.
The claim under test is not that mean divergence falls -- it is that the *spread*
across the population falls, because that spread is what an aggregate utility
number hides.
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
from sweep_cts_compression import js_divergence  # noqa: E402


def spectrum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a32, b32 = a.float(), b.float()
    qb, rb = torch.linalg.qr(b32, mode="reduced")
    qa, ra = torch.linalg.qr(a32.T, mode="reduced")
    return torch.linalg.svdvals(rb @ ra.T)


def kept_at(sigma: torch.Tensor, tau: float) -> int:
    energy = sigma.square()
    total = float(energy.sum())
    if total <= 0:
        return 0
    cum = torch.cumsum(energy, 0) / total
    return int(torch.searchsorted(cum, tau).item()) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, required=True,
                    help="cts_compression_sweep.json; supplies S and the adapter list")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--match-tau", type=float, default=0.90)
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="if >0, estimate beta by k-fold: each adapter's exponent is fitted "
                         "on the other folds only, so no adapter is allocated using an "
                         "exponent its own divergence helped determine.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="exponent in S^gamma sqrt(1-tau) <= eps.  gamma=1 equalises the "
                         "effective perturbation P; gamma = beta/2 for a measured "
                         "D ~ S^beta equalises the divergence instead.")
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    entries = json.loads(args.sweep.read_text())

    # Pass 1: spectra only.  No forward passes, so the whole pool is cheap.
    pool = []
    for e in entries:
        task = e["task"]
        local = Path(snapshot_download(f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{task}",
                                       local_dir=args.work / f"task{task:04d}"))
        weights = load_file(local / "adapter_model.safetensors")
        spectra = {}
        for a_key in sorted(k for k in weights if ".lora_A." in k):
            spectra[a_key] = spectrum(weights[a_key], weights[a_key.replace(".lora_A.", ".lora_B.")])
        pool.append({"entry": e, "local": local, "weights": weights, "spectra": spectra})
        print(f"spectra task{task:04d} ({len(spectra)} modules)", flush=True)

    def fit_beta(items) -> float:
        import statistics as _st
        key = f"e{round(args.match_tau * 100):02d}"
        xs = [math.log(i["entry"]["S"]) for i in items]
        ys = [math.log(i["entry"]["variants"][key]["d_js_mean"]) for i in items]
        mx, my = _st.mean(xs), _st.mean(ys)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

    # Fold assignment interleaves the S-sorted pool so every fold spans the range.
    order = sorted(range(len(pool)), key=lambda i: pool[i]["entry"]["S"])
    folds = args.cv_folds or 1
    for rank, idx in enumerate(order):
        pool[idx]["fold"] = rank % folds

    for f in range(folds):
        members = [p for p in pool if p["fold"] == f]
        if args.cv_folds:
            train = [p for p in pool if p["fold"] != f]
            gamma = fit_beta(train) / 2.0
        else:
            gamma = args.gamma
        # eps is matched to the budget of *this* fold, so a held-out adapter is
        # never sized against directions belonging to the training adapters.
        target = sum(sum(kept_at(s, args.match_tau) for s in m["spectra"].values())
                     for m in members)

        def total_at(eps: float) -> int:
            n = 0
            for m in members:
                tau = max(0.0, min(0.999999, 1.0 - (eps / m["entry"]["S"] ** gamma) ** 2))
                n += sum(kept_at(s, tau) for s in m["spectra"].values())
            return n

        lo, hi = 1e-8, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if total_at(mid) > target:
                lo = mid
            else:
                hi = mid
        for m in members:
            m["gamma"], m["eps"] = gamma, hi
        print(f"fold {f}: n={len(members)} gamma={gamma:.4f} eps={hi:.5f} "
              f"budget {target} -> {total_at(hi)}", flush=True)
    print(flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda").eval()

    peft_model, results = None, []
    for p in pool:
        e, local, weights = p["entry"], p["local"], p["weights"]
        gamma, eps = p["gamma"], p["eps"]
        tau = max(0.0, min(0.999999, 1.0 - (eps / e["S"] ** gamma) ** 2))
        out, kept_total, nominal_total = dict(weights), 0, 0
        for a_key in sorted(k for k in weights if ".lora_A." in k):
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            new_a, new_b, kept, _ = truncated_factors(weights[a_key], weights[b_key], tau)
            out[a_key], out[b_key] = new_a, new_b
            kept_total += kept
            nominal_total += weights[a_key].shape[0]
        dest = args.work / f"task{e['task']:04d}-adj"
        dest.mkdir(parents=True, exist_ok=True)
        save_file(out, dest / "adapter_model.safetensors")
        (dest / "adapter_config.json").write_text((local / "adapter_config.json").read_text())

        rows = load_dataset(e["dataset"], split="train")
        field = next(c for c in ("input", "text", "prompt") if c in rows.column_names)
        texts = [r for r in rows[field][: args.prompts * 3] if r and r.strip()][: args.prompts]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length)
        parts = [{k: v[i:i + args.chunk].to("cuda") for k, v in enc.items()}
                 for i in range(0, len(texts), args.chunk)]

        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(local), adapter_name="orig")
        else:
            peft_model.load_adapter(str(local), adapter_name="orig")
        peft_model.load_adapter(str(dest), adapter_name="adj")
        peft_model.eval()

        js_all, flips, seen = [], 0, 0
        for part in parts:
            mask = part["attention_mask"].bool()
            with torch.no_grad():
                peft_model.set_adapter("orig")
                ref = peft_model(**part)
                peft_model.set_adapter("adj")
                var = peft_model(**part)
            js_all.append(js_divergence(ref.logits, var.logits)[mask].cpu())
            flips += int((ref.logits.argmax(-1)[mask] != var.logits.argmax(-1)[mask]).sum())
            seen += int(mask.sum())
            del ref, var
        js = torch.cat(js_all)
        results.append({
            "adapter": e["adapter"], "S": e["S"], "tau": tau,
            "gamma": gamma, "eps": eps, "fold": p["fold"],
            "rank_frac": kept_total / max(nominal_total, 1),
            "d_js_mean": float(js.mean()), "argmax_flip": flips / max(seen, 1),
            "uniform_d_js_mean": e["variants"][f"e{round(args.match_tau*100):02d}"]["d_js_mean"],
            "uniform_rank_frac": e["variants"][f"e{round(args.match_tau*100):02d}"]["rank_frac"],
        })
        r = results[-1]
        print(f"task{e['task']:04d} S={r['S']:.4f} tau={tau:.4f} rank={r['rank_frac']:.2f} "
              f"(uniform {r['uniform_rank_frac']:.2f})  JS={r['d_js_mean']:.4f} "
              f"(uniform {r['uniform_d_js_mean']:.4f})  flip={r['argmax_flip']:.1%}", flush=True)

        for name in ("orig", "adj"):
            peft_model.delete_adapter(name)
        torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"cv_folds": args.cv_folds,
                                           "match_tau": args.match_tau,
                                           "adapters": results}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
