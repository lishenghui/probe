#!/usr/bin/env python3
"""Where does a compression error get attenuated -- or not?

The same injected weight perturbation (L_W ~ 0.30 at e90) reaches the model
output as 0.6% for video, 1.8-2.5% for the language models, and 30% for the VLA
policy.  The first three attenuate it by 12-42x; the VLA passes it through
essentially unchanged.  Since OpenVLA's backbone is a Llama-2 decoder like the
language models, the difference has to sit either in the backbone or in the
readout, and that is a question about *where along the network* the divergence
grows or shrinks.

So hook every adapted module and record how the relative divergence evolves with
depth:

    D_h(l) = || h_l^compressed - h_l^original || / || h_l^original ||

Reference selection differs by family: the language adapters are compared
against their uncompressed adapter, while the VLA corrections are applied to a
checkpoint that already carries the original update merged in, so there the
reference is the model with the adapter disabled.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def adapted_modules(model):
    """The base Linear inside each LoRA-injected module, in definition order."""
    found = []
    for name, module in model.named_modules():
        if hasattr(module, "base_layer") and hasattr(module, "lora_A"):
            found.append((name, module))
    return found


def depth_of(name: str) -> int:
    """Layer index if the module name carries one, else -1 (readout / vision)."""
    match = re.search(r"layers?\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--original", type=Path,
                        help="reference adapter; omit when the base already carries it")
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda"
    captured: dict[str, torch.Tensor] = {}
    scalars: dict[str, float] = {}
    reference: dict[str, torch.Tensor] = {}
    state = {"reducing": False}

    def make_hook(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if state["reducing"]:
                # Variant pass: the reference activation is already resident, so
                # collapse to the divergence here.  Shipping 439 activations to
                # host memory per forward is what pinned the GPU at 4% util.
                ref = reference.get(name)
                if ref is not None and ref.shape == tensor.shape:
                    d = (tensor.detach().float() - ref).norm() / ref.norm().clamp_min(1e-8)
                    scalars[name] = float(d)
            else:
                captured[name] = tensor.detach().float()
        return hook

    if args.observations:                     # VLA path
        from omegaconf import OmegaConf
        from peft import PeftModel
        from rlinf.models.embodiment.openvla_oft import rlinf as vla_factory

        cfg = OmegaConf.load(args.eval_config).rollout.model
        cfg.model_path = str(args.model)
        OmegaConf.set_struct(cfg, False)
        for key, value in (("center_crop", True), ("action_dim", 7),
                           ("num_action_chunks", 8), ("add_value_head", False),
                           ("trust_remote_code", True)):
            cfg.setdefault(key, value)
        cfg_fn = next(getattr(vla_factory, n) for n in dir(vla_factory)
                      if n.startswith("get_") and "config" in n)
        _, processor = cfg_fn(cfg)
        model = vla_factory.get_model(cfg).to(device).eval()
        model.input_processor = processor
        model = PeftModel.from_pretrained(model, str(args.variant))
        model.eval()

        steps = sorted(args.observations.glob("step_*.pt"))[: args.batches]
        payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in steps]

        def run(disabled, only=None):
            outs = []
            for i, record in enumerate(payloads):
                if only is not None and i != only:
                    continue
                env_obs = {k: (v.to(device) if torch.is_tensor(v) else v)
                           for k, v in record["env_obs"].items()}
                ctx = model.disable_adapter() if disabled else _null()
                with ctx, torch.no_grad():
                    res = model.predict_action_batch(
                        env_obs=env_obs, calculate_logprobs=False,
                        calculate_values=False, do_sample=False)
                act = res[0] if isinstance(res, tuple) else res
                outs.append((torch.as_tensor(act).detach().float().cpu(),
                             dict(captured)))
                captured.clear()
            return outs
    else:                                     # language / VLM path
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
            try:
                base = loader.from_pretrained(args.model, dtype=torch.bfloat16,
                                              trust_remote_code=True).to(device).eval()
                break
            except ValueError:
                continue
        model = PeftModel.from_pretrained(base, str(args.original), adapter_name="ref")
        model.load_adapter(str(args.variant), adapter_name="var")
        model.eval()

        texts = []
        for line in args.prompts.open():
            if line.strip():
                texts.append(json.loads(line)["text"])
            if len(texts) >= args.batches * 4:
                break
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=384)
        payloads = [{k: v[i:i + 4].to(device) for k, v in enc.items()}
                    for i in range(0, len(texts), 4)][: args.batches]

        def run(disabled, only=None):
            outs = []
            for i, part in enumerate(payloads):
                if only is not None and i != only:
                    continue
                model.set_adapter("ref" if disabled else "var")
                with torch.no_grad():
                    res = model(**part, output_hidden_states=True)
                mask = part["attention_mask"].bool()
                outs.append((res.hidden_states[-1][mask].detach().float().cpu(),
                             dict(captured)))
                captured.clear()
            return outs

    handles = [module.register_forward_hook(make_hook(name))
               for name, module in adapted_modules(model)]
    print(f"hooked {len(handles)} adapted modules", flush=True)

    # Compare one batch at a time: holding every hook output for every batch
    # is 439 modules x 8 batches of activations, which OOM-killed the first run.
    per_module: dict[str, list[float]] = {}
    final = []
    for index in range(len(payloads)):
        state["reducing"] = False
        ref_out, ref_h = run(disabled=True, only=index)[0]
        reference.clear()
        reference.update(ref_h)
        ref_h = None
        state["reducing"] = True
        scalars.clear()
        var_out, _ = run(disabled=False, only=index)[0]
        state["reducing"] = False
        final.append(float((var_out - ref_out).norm() / ref_out.norm().clamp_min(1e-8)))
        for name, value in scalars.items():
            per_module.setdefault(name, []).append(value)
        print(f"batch {index + 1}/{len(payloads)}: {len(scalars)} modules, "
              f"final D_h={final[-1]:.5f}", flush=True)
        reference.clear()
        captured.clear()
        del ref_out, var_out
        torch.cuda.empty_cache()
    for handle in handles:
        handle.remove()

    rows = [{"module": n, "depth": depth_of(n), "d_h": sum(v) / len(v)}
            for n, v in per_module.items()]
    rows.sort(key=lambda r: (r["depth"], r["module"]))
    by_depth: dict[int, list[float]] = {}
    for r in rows:
        by_depth.setdefault(r["depth"], []).append(r["d_h"])

    print(f"\n{args.family}: divergence by depth")
    print(f"{'depth':>6s} {'modules':>8s} {'D_h mean':>10s}")
    for depth in sorted(by_depth):
        vals = by_depth[depth]
        print(f"{depth:6d} {len(vals):8d} {sum(vals)/len(vals):10.5f}")
    print(f"\nfinal output divergence: {sum(final)/len(final):.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"family": args.family, "final_output_divergence": sum(final) / len(final),
         "modules": rows}, indent=2) + "\n")
    print(f"wrote {args.output}")


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
