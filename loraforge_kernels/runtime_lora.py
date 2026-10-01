"""Attach LoRA sidecars to plain ``nn.Linear`` layers as fused modules.

Runtime (non-merged) LoRA is usually bolted on with a forward hook that adds
``alpha * B(A(x))`` to the layer output.  That keeps the base layer untouched
but costs three extra passes over an ``[M, N]`` activation -- none of which
shrink when the adapter's rank does.  Swapping the ``nn.Linear`` for a
:class:`~loraforge_kernels.fused_linear.FusedLoRALinear` keeps the same
non-destructive semantics while letting the sidecar ride inside the base GEMM,
so its cost tracks the retained rank.
"""

from __future__ import annotations

import torch
from torch import nn

from .fused_linear import FusedLoRALinear


def resolve_parent(root: nn.Module, dotted_path: str) -> tuple[nn.Module, str]:
    """Split ``"a.b.c"`` into the module holding ``c`` and the attribute name."""
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def attach_fused_lora(
    root: nn.Module,
    dotted_path: str,
    down: torch.Tensor,
    up: torch.Tensor,
    alpha: float = 1.0,
) -> FusedLoRALinear:
    """Give the ``nn.Linear`` at ``dotted_path`` a fused LoRA sidecar.

    Calling this twice on the same path stacks the adapters by concatenating
    their factors, which is exact: ``a1*B1@A1 + a2*B2@A2 == [B1|B2] @ [a1*A1;a2*A2]``.
    One kernel then serves the whole stack at the summed rank.
    """
    parent, attr = resolve_parent(root, dotted_path)
    target = getattr(parent, attr)

    if isinstance(target, FusedLoRALinear):
        device, dtype = target.weight.device, target.weight.dtype
        # Fold the new adapter's alpha into its shrink factor so the stack can
        # share the single scale the layer already carries.
        folded = (alpha / target.scale) * down.detach().to(device=device, dtype=torch.float32)
        target.set_factors(
            torch.cat([target.lora_a, folded.to(dtype)], dim=0),
            torch.cat([target.lora_b, up.detach().to(device=device, dtype=dtype)], dim=1),
        )
        return target

    if not isinstance(target, nn.Linear):
        raise TypeError(f"LoRA target {dotted_path} is {type(target).__name__}, not nn.Linear")

    fused = FusedLoRALinear(target, down, up, scale=alpha)
    setattr(parent, attr, fused)
    return fused


def fused_lora_module_count(root: nn.Module) -> int:
    return sum(1 for module in root.modules() if isinstance(module, FusedLoRALinear))


def total_fused_rank(root: nn.Module) -> int:
    return sum(module.rank for module in root.modules() if isinstance(module, FusedLoRALinear))
