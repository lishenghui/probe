#!/usr/bin/env python3
"""Counterfactual strength intervention, plus the L_W needed for a joint fit.

The population result is a correlation: adapters with larger S are damaged more
by the same tau.  A reviewer can answer that S is a proxy for task difficulty --
hard tasks need big updates and are also fragile.  Scaling an adapter separates
the two.  For any lambda,

    dW -> lambda * dW    leaves the singular *ratios*, the cumulative energy
                         curve, and therefore the retained rank r_tau untouched,
                         while S -> lambda * S.

So one adapter, one task, one spectrum, one retained rank, and only the strength
moves.  We compare the lambda-scaled truncated adapter against the lambda-scaled
uncompressed one, so the reference moves with the intervention.

The same pass records L_l(tau) = ||dW - dW_tau||_F / ||dW||_F per adapter, the
exact adapter-relative residual.  Energy truncation keeps the smallest k with
E(k) >= tau, so the achieved residual is <= sqrt(1-tau) and varies by adapter;
the joint fit log D = a + beta log S + gamma log L needs the achieved value, not
the nominal bound.
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


def residual(a: torch.Tensor, b: torch.Tensor, tau: float) -> tuple[float, float, int, int]:
    """(kept energy, squared residual, kept rank, nominal rank) for one module."""
    a32, b32 = a.float(), b.float()
    qb, rb = torch.linalg.qr(b32, mode="reduced")
    qa, ra = torch.linalg.qr(a32.T, mode="reduced")
    sv = torch.linalg.svdvals(rb @ ra.T)
    e = sv.square()
    total = float(e.sum())
    if total <= 0:
        return 0.0, 0.0, 0, a32.shape[0]
    cum = torch.cumsum(e, 0) / total
    k = int(torch.searchsorted(cum, tau).item()) + 1
    return float(e[:k].sum()), float(e[k:].sum()), k, a32.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.95, 0.90])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--lambda-tau", type=float, default=0.90)
    ap.add_argument("--lambda-adapters", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1,
                    help="the lambda sweep is independent per adapter, so this splits "
                         "pass 2 across GPUs; pass 1 is spectra only and each shard "
                         "recomputes the same L_W table for free")
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--truncation-side", choices=("left", "right"), default="left",
                    help="SuperNI prompts end with the task instance, so the default "
                         "right truncation removes it: at max_length=320 that collapsed "
                         "20 of 30 pool adapters' 48 prompts to one shared preamble")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    entries = sorted(json.loads(args.sweep.read_text()), key=lambda r: r["S"])

    # Pass 1: exact adapter-relative residual L(tau) per adapter.  Spectra only.
    lw = []
    for e in entries:
        local = Path(snapshot_download(
            f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{e['task']}",
            local_dir=args.work / f"task{e['task']:04d}"))
        w = load_file(local / "adapter_model.safetensors")
        row = {"adapter": e["adapter"], "task": e["task"], "S": e["S"]}
        for tau in args.thresholds:
            kept = resid = 0.0
            k_tot = n_tot = 0
            for a_key in sorted(k for k in w if ".lora_A." in k):
                ke, re_, k, n = residual(w[a_key], w[a_key.replace(".lora_A.", ".lora_B.")], tau)
                kept += ke
                resid += re_
                k_tot += k
                n_tot += n
            label = f"e{round(tau * 100):02d}"
            row[f"L_{label}"] = math.sqrt(resid / (kept + resid)) if kept + resid > 0 else 0.0
            row[f"rank_frac_{label}"] = k_tot / max(n_tot, 1)
        lw.append(row)
        print(f"L_W task{e['task']:04d} S={e['S']:.4f} " +
              " ".join(f"{t}={row['L_' + t]:.4f}" for t in
                       (f"e{round(x*100):02d}" for x in args.thresholds)), flush=True)

    # Pass 2: the intervention, on adapters spread evenly over the S range.
    step = max(len(entries) // args.lambda_adapters, 1)
    chosen = entries[::step][: args.lambda_adapters]
    chosen = chosen[args.shard::args.shards]
    print(f"\nlambda sweep on {len(chosen)} adapters (shard {args.shard}/{args.shards}) "
          f"at tau={args.lambda_tau}\n", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.truncation_side = args.truncation_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dt).to("cuda").eval()

    peft_model, results = None, []
    for e in chosen:
        local = args.work / f"task{e['task']:04d}"
        w = load_file(local / "adapter_model.safetensors")
        a_keys = sorted(k for k in w if ".lora_A." in k)

        rows = load_dataset(e["dataset"], split="train")
        field = next(c for c in ("input", "text", "prompt") if c in rows.column_names)
        texts = [r for r in rows[field][: args.prompts * 3] if r and r.strip()][: args.prompts]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length)
        parts = [{k: v[i:i + args.chunk].to("cuda") for k, v in enc.items()}
                 for i in range(0, len(texts), args.chunk)]

        for lam in args.lambdas:
            # Scaling B scales dW; the SVD is homogeneous, so the retained rank
            # is identical for every lambda and only S moves.
            ref_w, var_w = dict(w), dict(w)
            kept_rank = 0
            for a_key in a_keys:
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                scaled_b = w[b_key].float() * lam
                ref_w[b_key] = scaled_b.to(w[b_key].dtype)
                new_a, new_b, k, _ = truncated_factors(w[a_key], scaled_b, args.lambda_tau)
                var_w[a_key], var_w[b_key] = new_a, new_b
                kept_rank += k
            for name, tensors in (("ref", ref_w), ("var", var_w)):
                d = args.work / f"task{e['task']:04d}-lam{lam}-{name}"
                d.mkdir(parents=True, exist_ok=True)
                save_file(tensors, d / "adapter_model.safetensors")
                (d / "adapter_config.json").write_text((local / "adapter_config.json").read_text())

            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    model, str(args.work / f"task{e['task']:04d}-lam{lam}-ref"), adapter_name="ref")
            else:
                peft_model.load_adapter(
                    str(args.work / f"task{e['task']:04d}-lam{lam}-ref"), adapter_name="ref")
            peft_model.load_adapter(
                str(args.work / f"task{e['task']:04d}-lam{lam}-var"), adapter_name="var")
            peft_model.eval()

            js_all, flips, seen = [], 0, 0
            for part in parts:
                mask = part["attention_mask"].bool()
                with torch.no_grad():
                    peft_model.set_adapter("ref")
                    r = peft_model(**part)
                    peft_model.set_adapter("var")
                    v = peft_model(**part)
                js_all.append(js_divergence(r.logits, v.logits)[mask].cpu())
                flips += int((r.logits.argmax(-1)[mask] != v.logits.argmax(-1)[mask]).sum())
                seen += int(mask.sum())
                del r, v
            js = torch.cat(js_all)
            results.append({"adapter": e["adapter"], "task": e["task"], "lambda": lam,
                            "S_effective": lam * e["S"], "S_base": e["S"],
                            "kept_rank": kept_rank, "d_js_mean": float(js.mean()),
                            "argmax_flip": flips / max(seen, 1)})
            print(f"task{e['task']:04d} lam={lam:4.2f} S_eff={lam*e['S']:.4f} "
                  f"rank={kept_rank} JS={float(js.mean()):.5f} "
                  f"flip={flips/max(seen,1):.1%}", flush=True)
            for name in ("ref", "var"):
                peft_model.delete_adapter(name)
            torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"L_W": lw, "lambda_tau": args.lambda_tau,
                                           "scaling": results}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
