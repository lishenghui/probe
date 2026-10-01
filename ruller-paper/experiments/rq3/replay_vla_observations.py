#!/usr/bin/env python3
"""Same-observation probing: how far does compression move a single action?

Closed-loop success drops 11.7% at e90, but a rollout confounds two things --
the per-step policy error and the amplification the environment applies to it.
Replaying the *original* policy's observations through each compressed variant
removes the second: nothing has diverged yet, so what is left is the local
error alone.

    D_a = || a_c - a_o || / || a_o ||

Compare against the same quantity measured for the other modalities (video
velocity 0.6-0.9%, VLM hidden state 2.5% at e90).  If D_a lands in that range
while success falls 11.7%, the loss is amplification, not a bigger local error.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def load_policy(model_path: Path, eval_config: Path, adapter: Path | None, device: str):
    """Build the policy through RLinf's own factory.

    The class takes action_dim / num_action_chunks / add_value_head /
    max_prompt_length that live in the eval config, so constructing it directly
    means duplicating those defaults; going through get_model keeps this probe
    aligned with whatever the rollout used.
    """
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openvla_oft import rlinf as vla_factory

    cfg = OmegaConf.load(eval_config)
    model_cfg = cfg.rollout.model
    model_cfg.model_path = str(model_path)
    OmegaConf.set_struct(model_cfg, False)
    model_cfg.setdefault("center_crop", True)
    model_cfg.setdefault("action_dim", 7)
    model_cfg.setdefault("num_action_chunks", 8)
    model_cfg.setdefault("add_value_head", False)
    model_cfg.setdefault("trust_remote_code", True)

    # The helper that builds config + processor is not exported under a stable
    # name across RLinf revisions, so find it by what it returns.
    cfg_fn = next(getattr(vla_factory, n) for n in dir(vla_factory)
                  if n.startswith("get_") and "config" in n and callable(getattr(vla_factory, n)))
    _, input_processor = cfg_fn(model_cfg)
    model = vla_factory.get_model(model_cfg).to(device).eval()
    model.input_processor = input_processor
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
        model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--variant", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    device = "cuda"
    steps = sorted(args.observations.glob("step_*.pt"))[: args.limit]
    if not steps:
        raise SystemExit(f"no observations under {args.observations}")
    print(f"{len(steps)} observation batches", flush=True)

    # Reference actions were recorded during the rollout that produced these
    # observations, so the original policy does not need re-running.
    reference = [torch.load(p, map_location="cpu", weights_only=False) for p in steps]

    rows = []
    print(f"\n{'variant':8s} {'D_a mean':>9s} {'D_a p90':>9s} {'D_a max':>9s} {'batches':>8s}")
    for adapter in args.variant:
        model = load_policy(args.model, args.eval_config, adapter, device)
        ratios = []
        for record in reference:
            env_obs = {k: (v.to(device) if torch.is_tensor(v) else v)
                       for k, v in record["env_obs"].items()}
            with torch.no_grad():
                out = model.predict_action_batch(
                    env_obs=env_obs, calculate_logprobs=False,
                    calculate_values=False, do_sample=False)
            actions = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(actions):
                actions = torch.as_tensor(actions)
            a_c = actions.detach().float().cpu().reshape(-1)
            a_o = record["actions"].float().reshape(-1)
            ratios.append(float((a_c - a_o).norm() / a_o.norm().clamp_min(1e-8)))
        del model
        torch.cuda.empty_cache()
        ratios.sort()
        stat = {"variant": adapter.name, "batches": len(ratios),
                "d_a_mean": statistics.fmean(ratios),
                "d_a_p90": ratios[int(0.9 * len(ratios))], "d_a_max": ratios[-1]}
        rows.append(stat)
        print(f"{adapter.name:8s} {stat['d_a_mean']:9.5f} {stat['d_a_p90']:9.5f} "
              f"{stat['d_a_max']:9.5f} {stat['batches']:8d}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
