"""Precision baselines for LoRA factors: quantize, dequantize, and count stored bytes.

Scales are per direction (each row of A, each column of B), symmetric, stored in
FP16. INT4 additionally groups each direction's long dimension into blocks of
`group` elements with one scale per block; two 4-bit values pack into one byte.
`fake_quantize` returns BF16 tensors equal to what a kernel would see after
dequantizing, so generation quality can be measured with the BF16 code path;
`stored_bytes` gives the packed size those factors would occupy.
"""
from __future__ import annotations

import torch

FORMATS = ('bf16', 'fp8', 'int8', 'int4')
SCALE_BYTES = 2
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _per_direction(t: torch.Tensor, side: str) -> torch.Tensor:
    # Directions are rows of A [r, d_in] and columns of B [d_out, r]; work on [r, n].
    return t if side == 'A' else t.t()


def _quantize_rows(x: torch.Tensor, fmt: str, group: int) -> torch.Tensor:
    x = x.float()
    if fmt == 'fp8':
        scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        return (x / scale).to(torch.float8_e4m3fn).float() * scale
    if fmt == 'int8':
        scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
        return (x / scale).round().clamp(-127, 127) * scale
    if fmt == 'int4':
        rows, n = x.shape
        pad = (-n) % group
        xp = torch.nn.functional.pad(x, (0, pad)).view(rows, -1, group)
        scale = xp.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / 7
        q = (xp / scale).round().clamp(-7, 7) * scale
        return q.view(rows, -1)[:, :n]
    raise ValueError(fmt)


def fake_quantize(state: dict, fmt: str, group: int = 128) -> dict:
    if fmt == 'bf16':
        return dict(state)
    out = {}
    for key, t in state.items():
        side = 'A' if key.endswith(('lora_A.weight', 'lora_down.weight')) else 'B'
        rows = _per_direction(t, side)
        q = _quantize_rows(rows, fmt, group)
        out[key] = (q if side == 'A' else q.t()).to(t.dtype).contiguous()
    return out


def stored_bytes(state: dict, fmt: str, group: int = 128) -> int:
    total = 0
    for key, t in state.items():
        side = 'A' if key.endswith(('lora_A.weight', 'lora_down.weight')) else 'B'
        r, n = _per_direction(t, side).shape
        if fmt == 'bf16':
            total += 2 * r * n
        elif fmt in ('fp8', 'int8'):
            total += r * n + SCALE_BYTES * r
        elif fmt == 'int4':
            total += (r * n + 1) // 2 + SCALE_BYTES * r * -(-n // group)
        else:
            raise ValueError(fmt)
    return total
