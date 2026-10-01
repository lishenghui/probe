#!/usr/bin/env python3
"""External validation of the calibration law on an independent adapter library.

LoraHub releases 196 task-specific LoRAs for google/flan-t5-large, all at r=16,
alpha=32, target modules {q, v}, alongside the FLAN-v2 data each was trained on.
It is independent of the Lots-of-LoRAs population in every respect that matters
here -- different authors, different base model, and encoder-decoder rather than
decoder-only -- while holding the adapter configuration fixed, which is what the
comparison needs.

The hypothesis is frozen before looking at this population: strength and spectral
loss should be complementary, and their product P = S * L_W should carry
essentially all of their joint predictive content.  We therefore report R^2 for
the four nested models rather than chasing a matching exponent; the response
curve may well be architecture-dependent while the coordinate is not.

Adapter selection is fixed in advance: sort every flan_t5_large-* repository by
name, seed 0, take a sample of --adapters.  No adapter is inspected first.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402
from sweep_cts_compression import js_divergence  # noqa: E402


def base_key(a_key: str) -> str:
    name = re.sub(r"\.lora_A\.(default\.)?weight$", ".weight", a_key)
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def spectrum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a32, b32 = a.float(), b.float()
    qb, rb = torch.linalg.qr(b32, mode="reduced")
    qa, ra = torch.linalg.qr(a32.T, mode="reduced")
    return torch.linalg.svdvals(rb @ ra.T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="google/flan-t5-large")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--adapters", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.95, 0.90])
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--max-source", type=int, default=384)
    ap.add_argument("--max-target", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32",
                    help="LoraHub adapters are ~20x weaker than the Mistral pool, so the "
                         "divergences land near 1e-5; bf16 logits do not resolve that.")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1.0],
                    help="scale each adapter before truncating.  lambda>1 extends the "
                         "population's strength range without altering any spectrum.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    api = HfApi()
    repos = sorted(m.id for m in api.list_models(author="lorahub")
                   if "flan_t5_large-" in m.id)
    stems = {f.rsplit(".json", 1)[0].split(":")[0]: f
             for f in (s.rfilename for s in api.dataset_info("lorahub/flanv2").siblings)
             if f.endswith(".json")}
    eligible = [r for r in repos if r.split("flan_t5_large-")[-1] in stems]
    rng = random.Random(0)
    chosen = sorted(rng.sample(eligible, min(args.adapters, len(eligible))))
    chosen = chosen[args.shard::args.shards]
    print(f"{len(repos)} repos, {len(eligible)} with data; shard {args.shard} takes "
          f"{len(chosen)} of {min(args.adapters, len(eligible))}", flush=True)

    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = T5ForConditionalGeneration.from_pretrained(args.base, dtype=dt).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(args.base)
    norms = {n: float(p.detach().float().norm()) for n, p in model.named_parameters()}
    print(f"base loaded, {len(norms)} weight tensors", flush=True)

    peft_model, results = None, []
    for repo in chosen:
        task = repo.split("flan_t5_large-")[-1]
        local = Path(snapshot_download(repo, local_dir=args.work / task))
        weights = load_file(local / "adapter_model.safetensors") \
            if (local / "adapter_model.safetensors").is_file() \
            else torch.load(local / "adapter_model.bin", map_location="cpu")
        a_keys = sorted(k for k in weights if ".lora_A." in k)

        cfg = json.loads((local / "adapter_config.json").read_text())
        scale = float(cfg["lora_alpha"]) / int(cfg["r"])

        # Global concatenated S and, per threshold, the achieved L_W, so that
        # P = S * L_W is the exact model-relative residual.
        num = den = 0.0
        spectra = {}
        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            w = norms.get(base_key(a_key))
            if w is None or b_key not in weights:
                continue
            sv = spectrum(weights[a_key], weights[b_key]) * scale
            spectra[a_key] = sv
            num += float(sv.square().sum())
            den += w * w
        if not spectra:
            print(f"skip {task}: no matched modules", flush=True)
            continue
        S = math.sqrt(num / den)

        rows = [json.loads(l) for l in open(
            hf_hub_download("lorahub/flanv2", stems[task], repo_type="dataset"))
            if l.strip()][: args.prompts]
        src = tok([r["inputs"] for r in rows], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_source)
        tgt = tok([r["targets"] for r in rows], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_target)
        parts = [({k: v[i:i + args.chunk].to("cuda") for k, v in src.items()},
                  {k: v[i:i + args.chunk].to("cuda") for k, v in tgt.items()})
                 for i in range(0, len(rows), args.chunk)]

        for lam in args.lambdas:
          record = {"adapter": task, "repo": repo, "S": S * lam, "S_base": S,
                    "lambda": lam, "modules": len(spectra), "variants": {}}
          scaled = {k: (v.float() * lam).to(v.dtype) if ".lora_B." in k else v
                    for k, v in weights.items()}
          for tau in args.thresholds:
            label = f"e{round(tau * 100):02d}"
            out, kept, resid, k_tot, n_tot = dict(scaled), 0.0, 0.0, 0, 0
            for a_key, sv in spectra.items():
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                new_a, new_b, k, _ = truncated_factors(scaled[a_key], scaled[b_key], tau)
                out[a_key], out[b_key] = new_a, new_b
                e = sv.square()
                kept += float(e[:k].sum())
                resid += float(e[k:].sum())
                k_tot += k
                n_tot += sv.numel()
            dest = args.work / f"{task}-lam{lam}-{label}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text((local / "adapter_config.json").read_text())

            ref_dir = args.work / f"{task}-lam{lam}-ref"
            ref_dir.mkdir(parents=True, exist_ok=True)
            save_file(scaled, ref_dir / "adapter_model.safetensors")
            (ref_dir / "adapter_config.json").write_text(
                (local / "adapter_config.json").read_text())
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(model, str(ref_dir), adapter_name="orig")
            else:
                peft_model.load_adapter(str(ref_dir), adapter_name="orig")
            peft_model.load_adapter(str(dest), adapter_name="var")
            peft_model.eval()

            js_all, flips, seen = [], 0, 0
            for s_part, t_part in parts:
                labels = t_part["input_ids"].clone()
                mask = t_part["attention_mask"].bool()
                labels[~mask] = -100
                with torch.no_grad():
                    peft_model.set_adapter("orig")
                    ref = peft_model(input_ids=s_part["input_ids"],
                                     attention_mask=s_part["attention_mask"], labels=labels)
                    peft_model.set_adapter("var")
                    var = peft_model(input_ids=s_part["input_ids"],
                                     attention_mask=s_part["attention_mask"], labels=labels)
                js_all.append(js_divergence(ref.logits, var.logits)[mask].cpu())
                flips += int((ref.logits.argmax(-1)[mask] != var.logits.argmax(-1)[mask]).sum())
                seen += int(mask.sum())
                del ref, var
            js = torch.cat(js_all)
            record["variants"][label] = {
                "L_W": math.sqrt(resid / (kept + resid)) if kept + resid > 0 else 0.0,
                "rank_frac": k_tot / max(n_tot, 1),
                "d_js_mean": float(js.mean()), "argmax_flip": flips / max(seen, 1),
                "positions": seen,
            }
            for name in ("orig", "var"):
                peft_model.delete_adapter(name)
            torch.cuda.empty_cache()
          results.append(record)
          line = "  ".join(f"{k}: L={v['L_W']:.3f} JS={v['d_js_mean']:.6f}"
                           for k, v in record["variants"].items())
          print(f"{task[:30]:32s} lam={lam:5.1f} S={S*lam:.4f}  {line}", flush=True)
          args.output.parent.mkdir(parents=True, exist_ok=True)
          args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(results)} adapters)")


if __name__ == "__main__":
    main()
