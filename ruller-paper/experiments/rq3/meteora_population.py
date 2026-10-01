#!/usr/bin/env python3
"""External validation on the MeteoRA adapter population (Llama-3-8B).

MeteoRA releases 28 task-specific LoRAs trained separately on BIG-Bench and
related tasks for one Llama-3-8B backbone, together with the per-task data.  It
is independent of the Lots-of-LoRAs population -- different authors, different
base model, r=8 rather than 16, seven target modules rather than seven on a
different architecture -- while still holding base, rank, and recipe fixed
within itself, which is what the comparison needs.

The hypothesis is frozen before this population is examined: strength and
spectral loss should be complementary, and their product P = S * L_W should
carry essentially all of their joint predictive content.  We therefore report
R^2 for the four nested models and the interval on a - b, not a matching
exponent; the response curve may be architecture-dependent while the coordinate
is not.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402
from sweep_cts_compression import js_divergence  # noqa: E402

REPO = "NJUDeepEngine/MeteoRA-llama3-8b"
DATA = "NJUDeepEngine/meteora_dataset"
PREFIX = "llama3_8b_peft"


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
    ap.add_argument("--base", type=str, default="NousResearch/Meta-Llama-3-8B")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.95, 0.90])
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                    help="this population's divergences sit near 1e-4; bf16 logits put a "
                         "floor under that, which suppresses the fitted R^2.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi()
    files = [s.rfilename for s in api.model_info(REPO).siblings]
    tasks = sorted({f.split("/")[1] for f in files
                    if f.startswith(f"{PREFIX}/") and f.count("/") >= 2})
    mine = tasks[args.shard::args.shards]
    print(f"{len(tasks)} tasks; shard {args.shard}/{args.shards} takes {len(mine)}: {mine}",
          flush=True)

    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dt).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    norms = {n: float(p.detach().float().norm()) for n, p in model.named_parameters()}
    print(f"base loaded, {len(norms)} weight tensors", flush=True)

    peft_model, results = None, []
    for task in mine:
        local = args.work / task
        local.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            src = hf_hub_download(REPO, f"{PREFIX}/{task}/{name}")
            (local / name).write_bytes(Path(src).read_bytes())
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
        if not spectra:
            print(f"skip {task}: no matched modules", flush=True)
            continue
        S = math.sqrt(num / den)

        rows = [json.loads(l) for l in
                open(hf_hub_download(DATA, f"{task}/test.jsonl", repo_type="dataset"))
                if l.strip()]
        texts = [r["prompt"] for r in rows[: args.prompts * 3]
                 if r.get("prompt", "").strip()][: args.prompts]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length)
        parts = [{k: v[i:i + args.chunk].to("cuda") for k, v in enc.items()}
                 for i in range(0, len(texts), args.chunk)]

        record = {"adapter": task, "S": S, "modules": len(spectra), "rank": rank,
                  "prompts": len(texts), "variants": {}}
        for tau in args.thresholds:
            label = f"e{round(tau * 100):02d}"
            out, kept, resid, k_tot, n_tot = dict(weights), 0.0, 0.0, 0, 0
            for a_key, sv in spectra.items():
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                new_a, new_b, k, _ = truncated_factors(weights[a_key], weights[b_key], tau)
                out[a_key], out[b_key] = new_a, new_b
                e = sv.square()
                kept += float(e[:k].sum())
                resid += float(e[k:].sum())
                k_tot += k
                n_tot += sv.numel()
            dest = args.work / f"{task}-{label}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text((local / "adapter_config.json").read_text())

            if peft_model is None:
                peft_model = PeftModel.from_pretrained(model, str(local), adapter_name="orig")
            else:
                peft_model.load_adapter(str(local), adapter_name="orig")
            peft_model.load_adapter(str(dest), adapter_name="var")
            peft_model.eval()

            js_all, flips, seen = [], 0, 0
            for part in parts:
                mask = part["attention_mask"].bool()
                with torch.no_grad():
                    peft_model.set_adapter("orig")
                    ref = peft_model(**part)
                    peft_model.set_adapter("var")
                    var = peft_model(**part)
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
        print(f"{task[:34]:36s} S={S:.4f}  {line}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(results)} adapters)")


if __name__ == "__main__":
    main()
