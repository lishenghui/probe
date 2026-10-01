"""Dual-stream overlapped execution of Base Linear GEMM and LoRA branch."""

from __future__ import annotations

import torch
from .fused_lora import fused_lora
from .splitk_lora import splitk_persistent_fused_lora


class DualStreamOverlappedLinear(torch.nn.Module):
    """
    Overlaps base model Linear GEMM (stream 0) and LoRA branch (stream 1)
    to hide small-rank LoRA latency entirely in the shadow of base GEMM.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False, dtype=torch.float16, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype) / in_features**0.5)
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features, device=device, dtype=dtype) / in_features**0.5)
        self.lora_B = torch.nn.Parameter(torch.randn(out_features, rank, device=device, dtype=dtype) / max(rank, 1)**0.5)
        self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None

        # Dedicated streams and synchronization events
        self.stream_lora = torch.cuda.Stream(device=device)
        self.event_input_ready = torch.cuda.Event()
        self.event_lora_done = torch.cuda.Event()

    def forward(self, x: torch.Tensor, scale: float = 1.0, use_splitk: bool = False) -> torch.Tensor:
        """
        Overlapped forward:
        Stream 0 (default): y_base = x @ W.T
        Stream 1 (lora):    delta  = fused_lora(x, A, B)
        Join:               y_base += delta
        """
        current_stream = torch.cuda.current_stream()
        
        # 1. Record that input x is ready
        self.event_input_ready.record(current_stream)

        # 2. Launch LoRA branch on concurrent stream
        with torch.cuda.stream(self.stream_lora):
            self.stream_lora.wait_event(self.event_input_ready)
            if use_splitk:
                delta = splitk_persistent_fused_lora(x, self.lora_A, self.lora_B, scale=scale)
            else:
                delta = fused_lora(x, self.lora_A, self.lora_B, scale=scale)
            self.event_lora_done.record(self.stream_lora)

        # 3. Launch Base GEMM on main stream
        y_base = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y_base += self.bias

        # 4. Wait for LoRA branch and add in-place
        current_stream.wait_event(self.event_lora_done)
        y_base.add_(delta)

        return y_base
