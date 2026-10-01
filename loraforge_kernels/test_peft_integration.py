#!/usr/bin/env python3
"""Correctness checks for PEFT decode selection and grouped MLP execution."""

import copy

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from loraforge_kernels.peft_integration import enable_loraforge_peft, peft_plan_report
from loraforge_kernels import tiled_lora


class ToyMLP(nn.Module):
    def __init__(self, width=256, hidden=512):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.up_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def main():
    torch.manual_seed(7)
    base = ToyMLP().to(device="cuda", dtype=torch.bfloat16).eval()
    cfg = LoraConfig(r=16, lora_alpha=16, target_modules=["gate_proj", "up_proj", "down_proj"])
    reference = get_peft_model(base, cfg).eval()
    candidate = copy.deepcopy(reference)
    patched = enable_loraforge_peft(candidate, enable_grouping=True)
    assert patched == 3
    assert candidate._loraforge_grouped_mlp_count == 1

    for rows in (1, 8, 128):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            want = reference(x)
            got = candidate(x)
        rel = float((got.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-6))
        print(f"rows={rows} rel_err={rel:.3e}")
        assert rel < 4e-2
    report = peft_plan_report(candidate)
    print(report)
    assert report["grouped_mlp_modules"] == 1

    # Output-tiled geometry must match the direct sidecar for decode shapes.
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(13, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 13, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    want = y + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)
    got = tiled_lora(x, a, b, y)
    rel = float((got.float() - want.float()).abs().max() / want.float().abs().max())
    print(f"tiled rel_err={rel:.3e}")
    assert rel < 4e-2


if __name__ == "__main__":
    main()
