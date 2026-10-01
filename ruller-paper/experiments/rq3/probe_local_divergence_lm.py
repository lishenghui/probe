#!/usr/bin/env python3
"""Prediction-space local divergence for the language / vision-language model.

Teacher forced: the original and the compressed adapter see the identical
prefix, so nothing has diverged yet and what is measured is purely the one-step
effect of compression.

    D_JS(t) = JS( p_o(. | y_<t), p_c(. | y_<t) )

reported alongside the relative change of the final hidden state, which is the
same quantity the diffusion probe measures for velocity.  Free-running
divergence is a separate question and needs generation; this is the local half.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Per-position Jensen-Shannon divergence in nats, computed in log space."""
    p = torch.log_softmax(p_logits.float(), dim=-1)
    q = torch.log_softmax(q_logits.float(), dim=-1)
    m = torch.logaddexp(p, q) - torch.log(torch.tensor(2.0, device=p.device))
    kl_pm = (p.exp() * (p - m)).sum(-1)
    kl_qm = (q.exp() * (q - m)).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--variant", type=Path, nargs="+", required=True)
    parser.add_argument("--prompts", type=Path, help="jsonl with a 'text' field")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Qwen3-VL is a vision-language model: it has no AutoModelForCausalLM entry,
    # but text-only batches run through the same decoder.
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            base = loader.from_pretrained(
                args.model, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
            print(f"loaded with {loader.__name__}", flush=True)
            break
        except ValueError:
            continue
    else:
        raise SystemExit("no AutoModel class accepted this checkpoint")

    texts = []
    if args.prompts and args.prompts.exists():
        for line in args.prompts.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            for field in ("text", "prompt", "instruction"):
                if isinstance(rec.get(field), str) and len(rec[field]) > 40:
                    texts.append(rec[field])
                    break
            if len(texts) >= args.limit:
                break
    if not texts:
        raise SystemExit("no prompts found; pass --prompts with a 'text' field")
    print(f"{len(texts)} prompts", flush=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    encoded = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length)

    def with_adapters(variant):
        """Both adapters on one model, selected by name.

        Wrapping `base` twice does not give two models: PEFT injects the LoRA
        layers in place, so the second wrapper silently rebinds the same modules
        and every divergence comes out as exactly zero.
        """
        model = PeftModel.from_pretrained(base, str(args.original), adapter_name="ref")
        model.load_adapter(str(variant), adapter_name="var")
        return model.eval()

    def chunks():
        """Prompts in small groups: a full [64, 512, 151k] logits tensor is 20 GB,
        which is what OOM-killed the first attempt."""
        for start in range(0, encoded["input_ids"].shape[0], args.chunk):
            stop = start + args.chunk
            yield {k: v[start:stop].to(device) for k, v in encoded.items()}

    rows = []
    print(f"\n{'variant':8s} {'D_JS mean':>10s} {'D_JS p90':>9s} {'D_hidden':>10s} {'argmax flip':>12s}")
    for path in args.variant:
        model = with_adapters(path)
        js_l, dh_l, flips, total = [], [], 0, 0
        with torch.no_grad():
            for part in chunks():
                mask = part["attention_mask"].bool()
                model.set_adapter("ref")
                ref = model(**part, output_hidden_states=True)
                model.set_adapter("var")
                var = model(**part, output_hidden_states=True)
                js = js_divergence(ref.logits, var.logits)[mask]
                rh, vh = ref.hidden_states[-1].float(), var.hidden_states[-1].float()
                dh = ((vh - rh).norm(dim=-1) / rh.norm(dim=-1).clamp_min(1e-8))[mask]
                flips += int((ref.logits.argmax(-1) != var.logits.argmax(-1))[mask].sum())
                total += int(mask.sum())
                js_l += js.float().cpu().tolist()
                dh_l += dh.float().cpu().tolist()
                del ref, var
        model.unload()
        del model
        torch.cuda.empty_cache()
        flip = flips / max(total, 1)
        dh = torch.tensor(dh_l)
        js_l = sorted(js_l)
        stat = {"variant": path.name, "js_mean": statistics.fmean(js_l),
                "js_p90": js_l[int(0.9 * len(js_l))],
                "hidden_rel": float(dh.mean()), "argmax_flip": float(flip),
                "positions": len(js_l)}
        rows.append(stat)
        print(f"{path.name:8s} {stat['js_mean']:10.5f} {stat['js_p90']:9.5f} "
              f"{stat['hidden_rel']:10.5f} {stat['argmax_flip']:11.3%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
