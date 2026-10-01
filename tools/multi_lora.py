"""Multi-adapter LoRA for the diffusers Wan transformer without PEFT.

Each targeted nn.Linear is wrapped by MultiLoRALinear, which keeps, per resident
adapter, its own compact A [r, d_in] and B [d_out, r] (any per-module rank, no
zero padding). A forward pass can apply one adapter to the whole batch or a
different adapter per batch sample (heterogeneous batch). Scale is 1, matching
alpha-free ComfyUI/Remade files.

Loading an adapter is only tensor copies into the wrapped layers (no module
injection), so swap cost is set by bytes moved; removing one frees its tensors.
"""
from __future__ import annotations

from collections import defaultdict
import re

import torch
from torch import nn

# Comfy/original Wan key -> diffusers WanTransformer3DModel module path (per block).
_MAP = {
    'self_attn.q': 'attn1.to_q', 'self_attn.k': 'attn1.to_k', 'self_attn.v': 'attn1.to_v',
    'self_attn.o': 'attn1.to_out.0', 'cross_attn.q': 'attn2.to_q', 'cross_attn.k': 'attn2.to_k',
    'cross_attn.v': 'attn2.to_v', 'cross_attn.o': 'attn2.to_out.0', 'cross_attn.k_img': 'attn2.add_k_proj',
    'cross_attn.v_img': 'attn2.add_v_proj', 'ffn.0': 'ffn.net.0.proj', 'ffn.2': 'ffn.net.2',
}
_KEY = re.compile(r'^(?:diffusion_model\.)?blocks\.(\d+)\.(.+)\.lora_(A|B|down|up)\.weight$')


def diffusers_name(key):
    match = _KEY.match(key)
    if not match or match.group(2) not in _MAP:
        return None, None
    side = 'A' if match.group(3) in ('A', 'down') else 'B'
    return f'blocks.{match.group(1)}.{_MAP[match.group(2)]}', side


class MultiLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.A: dict[str, torch.Tensor] = {}
        self.B: dict[str, torch.Tensor] = {}
        self.route = None  # None | adapter name | list of names (one per batch sample)

    def forward(self, x):
        y = self.base(x)
        route = self.route
        if route is None:
            return y
        if isinstance(route, str):
            if route in self.A:
                y = y + (x @ self.A[route].t()) @ self.B[route].t()
            return y
        if len(route) != x.shape[0]:
            raise ValueError(f'route has {len(route)} entries for batch {x.shape[0]}')
        groups = defaultdict(list)
        for i, name in enumerate(route):
            if name is not None and name in self.A:
                groups[name].append(i)
        if not groups:
            return y
        y = y.clone()
        for name, rows in groups.items():
            index = torch.tensor(rows, device=x.device)
            xs = x.index_select(0, index)
            y.index_add_(0, index, (xs @ self.A[name].t()) @ self.B[name].t())
        return y


class MultiLoRA:
    """Owns the wrapped layers of one transformer and the set of resident adapters."""

    def __init__(self, transformer):
        self.layers: dict[str, MultiLoRALinear] = {}
        for path in sorted({f'blocks.{i}.{m}' for i in range(len(transformer.blocks)) for m in _MAP.values()}):
            parent_path, _, child = path.rpartition('.')
            parent = transformer.get_submodule(parent_path)
            base = getattr(parent, child) if not child.isdigit() else parent[int(child)]
            if base is None:
                continue  # e.g. T2V blocks have no add_k_proj
            wrapped = MultiLoRALinear(base)
            if child.isdigit():
                parent[int(child)] = wrapped
            else:
                setattr(parent, child, wrapped)
            self.layers[path] = wrapped
        self.adapters: dict[str, dict] = {}

    def add(self, name, state, device='cuda', dtype=torch.bfloat16):
        """Copy one adapter's factors in. `state` may live on CPU (pinned for async copies)."""
        if name in self.adapters:
            raise ValueError(f'{name} already resident')
        pairs = defaultdict(dict)
        for key, tensor in state.items():
            path, side = diffusers_name(key)
            if path is not None:
                pairs[path][side] = tensor
        loaded, nbytes, directions = 0, 0, 0
        for path, factors in pairs.items():
            layer = self.layers[path]
            a = factors['A'].to(device=device, dtype=dtype, non_blocking=True)
            b = factors['B'].to(device=device, dtype=dtype, non_blocking=True)
            if a.shape[1] != layer.base.in_features or b.shape[0] != layer.base.out_features:
                raise ValueError(f'{name}:{path} shape {tuple(b.shape)}x{tuple(a.shape)} does not fit '
                                 f'{layer.base.out_features}x{layer.base.in_features}')
            layer.A[name], layer.B[name] = a, b
            loaded += 1
            nbytes += a.numel() * a.element_size() + b.numel() * b.element_size()
            directions += a.shape[0]
        expected = sum(1 for k in state if diffusers_name(k)[1] == 'A')
        if loaded == 0 or loaded != expected or len(state) != 2 * expected:
            raise ValueError(f'{name}: mapped {loaded} modules of {expected}; {len(state)} tensors in file')
        self.adapters[name] = dict(modules=loaded, bytes=nbytes, directions=directions)
        return self.adapters[name]

    def remove(self, name):
        for layer in self.layers.values():
            layer.A.pop(name, None)
            layer.B.pop(name, None)
        self.adapters.pop(name)

    def route(self, route):
        for layer in self.layers.values():
            layer.route = route
