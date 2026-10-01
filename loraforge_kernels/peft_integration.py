"""Inference-only PEFT integration for the rank-proportional LoRA path."""

from __future__ import annotations

import os
import statistics
import types
from collections import Counter

import torch
from peft.tuners.lora.layer import Linear as LoraLinear

from .fused_linear import MIN_GEMM_ROWS, _time_ms, build_plan
from .decode_linear import build_decode_linear
from .fused_lora import fused_lora, grouped_lora, tiled_lora
from .splitk_lora import splitk_persistent_fused_lora


_DECODE_CHOICE_CACHE = {}
_GROUPED_MLP_CHOICE_CACHE = {}


def _inactive_dropout(dropout):
    """Training dropout is an identity after eval(), regardless of saved p."""
    return isinstance(dropout, torch.nn.Identity) or (
        isinstance(dropout, torch.nn.Dropout) and not dropout.training
    )


def _sidecar(module, name):
    """Return (A, B, scale) for one active adapter, or None if unsupported."""
    if name not in module.lora_A or module.use_dora.get(name, False):
        return None
    a = module.lora_A[name].weight
    b = module.lora_B[name].weight
    if a.ndim != 2 or b.ndim != 2:
        return None
    return a, b, float(module.scaling[name])


def _build_plan(module, x, rows):
    """Resolve everything that does not change between steps, once."""
    declined = (module._active_adapter, rows, None)
    active = list(module.active_adapters)
    if len(active) != 1:
        return declined
    name = active[0]
    sidecar = _sidecar(module, name)
    if sidecar is None:
        return declined
    a, b, scale = sidecar
    base_layer = module.base_layer

    if not (
        isinstance(base_layer, torch.nn.Linear)
        and _inactive_dropout(module.lora_dropout[name])
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.dtype == base_layer.weight.dtype
        and a.dtype == x.dtype
        and b.dtype == x.dtype
    ):
        return declined

    if a.shape[0] == 0 or scale == 0:
        module._loraforge_decode_choice = "base"
        return module._active_adapter, rows, base_layer.forward

    probe = x.reshape(-1, x.shape[-1])
    if not probe.is_contiguous():
        probe = probe.contiguous()
    if rows < MIN_GEMM_ROWS:
        run = _build_decode_plan(module, probe, a, b, scale)
        out_features = base_layer.out_features

        def call(x):
            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            return run(x2).reshape(*shape[:-1], out_features)

        return module._active_adapter, rows, call
    run = build_plan(probe, base_layer.weight, base_layer.bias, a, b, scale,
                     base_layer=base_layer)
    out_features = base_layer.out_features

    def call(x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        return run(x2).reshape(*shape[:-1], out_features)

    # Guard on peft's raw attribute (str or list) so the check is one compare.
    return module._active_adapter, rows, call


def _cuda_median_us(fn, warmup=5, repeats=15):
    """Tune for eager launches or graph replay, matching the serving mode.

    Events around Python include host launch gaps. That is relevant to eager
    serving, but graph serving must measure GPU work without those gaps.
    """
    if os.environ.get("LORAFORGE_TUNE_MODE", "eager") != "graph":
        return _time_ms(fn, warmup=warmup, reps=repeats) * 1000.0
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    inner = 20
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    samples = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / inner)
    return statistics.median(samples)


def _build_decode_plan(module, probe, a, b, scale):
    """Choose the small-row path instead of forcing the one-CTA row kernel.

    Decode shapes are unusually sensitive to GPU geometry: the persistent row
    kernel is excellent for some (K,N,r) triples but launches only M programs,
    so at M=1--8 it can underfill a GH200 and lose even to two ordinary GEMMs.
    Benchmark eligible implementations once per shape and keep the winner.
    Full-model measurements must still check cache and scheduling effects.
    """
    base_layer = module.base_layer

    def direct(x):
        y = base_layer(x)
        z = torch.nn.functional.linear(x, a)
        return y.add_(torch.nn.functional.linear(z, b), alpha=scale)

    def row(x):
        y = base_layer(x)
        return fused_lora(x, a, b, y, scale)

    # Keep PEFT's own implementation in the race. Library overhead and dtype
    # handling can still beat a custom kernel for tiny decode shapes.
    candidates = {
        "peft": lambda x: module._loraforge_original_forward(x),
        "direct": direct,
    }
    # One launch includes the base projection. The shrink is recomputed per
    # output tile, so restrict the candidate to genuinely small decode shapes.
    if (probe.shape[0] <= 8 and a.shape[0] <= 16 and probe.shape[1] <= 4096
            and (probe.shape[0] == 1 or b.shape[0] <= 4096)
            and os.environ.get("LORAFORGE_ENABLE_DECODE_LINEAR", "1") == "1"):
        for tile in (8, 16, 32):
            candidates[f"decode{tile}"] = build_decode_linear(
                base_layer.weight, base_layer.bias, a, b, scale, block_n=tile,
            )
    if a.shape[0] <= 64:
        candidates["row"] = row
        # Output-tiled fusion wins most isolated shapes but can regress whole
        # models through cache pressure; keep it experimental until a model-
        # level selector is available.
        if os.environ.get("LORAFORGE_ENABLE_TILED", "0") == "1":
            candidates["tiled"] = lambda x: tiled_lora(x, a, b, base_layer(x), scale)
    forced = os.environ.get("LORAFORGE_DECODE_VARIANT", "").strip().lower()
    if forced in candidates:
        module._loraforge_decode_choice = forced
        module._loraforge_decode_timings_us = {"forced": True}
        return candidates[forced]

    signature = (
        probe.device.type, probe.device.index, probe.dtype, probe.shape[0],
        probe.shape[1], b.shape[0], a.shape[0], base_layer.bias is not None,
        tuple(candidates),
        os.environ.get("LORAFORGE_TUNE_MODE", "eager"),
    )
    cached = _DECODE_CHOICE_CACHE.get(signature)
    if cached in candidates:
        module._loraforge_decode_choice = cached
        module._loraforge_decode_timings_us = {"cache_hit": True}
        return candidates[cached]

    timings = {}
    for name, candidate in candidates.items():
        try:
            timings[name] = _cuda_median_us(lambda fn=candidate: fn(probe))
        except Exception:
            continue
    winner = min(timings, key=timings.get) if timings else "peft"
    _DECODE_CHOICE_CACHE[signature] = winner
    module._loraforge_decode_choice = winner
    module._loraforge_decode_timings_us = timings
    return candidates[winner]


def _loraforge_linear_forward(self, x: torch.Tensor, *args, **kwargs):
    """Drop-in inference forward for ``peft.tuners.lora.layer.Linear``.

    A plan is resolved once per (adapter, row count) and then reused, because a
    diffusion step calls this once per adapted linear per CFG pass -- thousands
    of times per generation, where the dispatch itself was costing more than the
    sidecar arithmetic.
    """
    if args or kwargs or self.training or self.merged or self.disable_adapters:
        return self._loraforge_original_forward(x, *args, **kwargs)

    rows = x.numel() // x.shape[-1]
    plan = getattr(self, "_loraforge_plan", None)
    if (plan is None or plan[0] != self._active_adapter or plan[1] != rows
            or plan[3] != self.scaling or plan[4] != (x.dtype, x.device)):
        cache = self._loraforge_plans
        key = (tuple(self.active_adapters), rows, tuple(self.scaling.items()), x.dtype, x.device)
        plan = cache.get(key)
        if plan is None:
            adapter, plan_rows, call = _build_plan(self, x, rows)
            # Snapshot mutable adapter/scaling state. A zero-scale base-only
            # plan must not survive PEFT's subsequent set_scale().
            adapter = adapter.copy() if isinstance(adapter, list) else adapter
            plan = (adapter, plan_rows, call, self.scaling.copy(), (x.dtype, x.device))
            # Keep the common prefill + decode pair. Repacking the entire base
            # on each request costs bandwidth and changes graph input pointers.
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            cache[key] = plan
        self._loraforge_plan = plan
    call = plan[2]
    if call is not None:
        return call(x)
    return self._loraforge_original_forward(x)


def _patch_grouped_attention(attn_module) -> bool:
    """Group Q, K, V projections into a single grouped_lora kernel launch."""
    q_proj = getattr(attn_module, "q_proj", None)
    k_proj = getattr(attn_module, "k_proj", None)
    v_proj = getattr(attn_module, "v_proj", None)

    if not (isinstance(q_proj, LoraLinear) and isinstance(k_proj, LoraLinear) and isinstance(v_proj, LoraLinear)):
        return False

    name = list(q_proj.active_adapters)[0] if q_proj.active_adapters else "default"
    if name not in q_proj.lora_A or name not in k_proj.lora_A or name not in v_proj.lora_A:
        return False

    rq, rk, rv = q_proj.lora_A[name].weight.shape[0], k_proj.lora_A[name].weight.shape[0], v_proj.lora_A[name].weight.shape[0]
    if not (rq == rk == rv and rq <= 32):
        return False

    r = rq
    # Pre-pack A and B for fast grouped execution
    a_q = q_proj.lora_A[name].weight
    a_k = k_proj.lora_A[name].weight
    a_v = v_proj.lora_A[name].weight
    a_qkv = torch.cat([a_q, a_k, a_v], dim=0).contiguous()

    b_q = q_proj.lora_B[name].weight
    b_k = k_proj.lora_B[name].weight
    b_v = v_proj.lora_B[name].weight
    b_qkv = torch.stack([b_q, b_k, b_v], dim=0).contiguous()

    scale = float(q_proj.scaling[name])

    attn_module._qkv_packed = (a_qkv, b_qkv, scale, r)
    return True


def _patch_grouped_mlp(mlp_module) -> bool:
    """Group Gate and Up sidecars into one decode kernel launch.

    Transformer MLPs evaluate ``gate_proj(x)`` immediately before ``up_proj(x)``.
    The gate call computes both base projections and both LoRA updates, returns
    gate, and caches up until the following call. Large-row paths keep their
    individually autotuned GEMM plans; grouping is only for launch-bound decode.
    """
    # Read direct children only. PEFT wrapper modules delegate ``__getattr__``
    # to their wrapped model, which would otherwise patch the same projections
    # once at every wrapper level.
    gate_proj = mlp_module._modules.get("gate_proj")
    up_proj = mlp_module._modules.get("up_proj")

    if getattr(mlp_module, "_loraforge_grouped_mlp", False):
        return False

    if not (isinstance(gate_proj, LoraLinear) and isinstance(up_proj, LoraLinear)):
        return False

    name = list(gate_proj.active_adapters)[0] if gate_proj.active_adapters else "default"
    if name not in gate_proj.lora_A or name not in up_proj.lora_A:
        return False

    rg, ru = gate_proj.lora_A[name].weight.shape[0], up_proj.lora_A[name].weight.shape[0]
    r = max(rg, ru)
    if r > 32:
        return False

    a_g = gate_proj.lora_A[name].weight
    a_u = up_proj.lora_A[name].weight
    b_g = gate_proj.lora_B[name].weight
    b_u = up_proj.lora_B[name].weight
    if not (
        _inactive_dropout(gate_proj.lora_dropout[name])
        and _inactive_dropout(up_proj.lora_dropout[name])
        and a_g.dtype == gate_proj.base_layer.weight.dtype
        and a_u.dtype == gate_proj.base_layer.weight.dtype
        and b_g.dtype == gate_proj.base_layer.weight.dtype
        and b_u.dtype == gate_proj.base_layer.weight.dtype
    ):
        return False

    # Per-matrix energy truncation produces heterogeneous ranks. Pad only this
    # local pair to its maximum rank, and absorb each projection's scale into B
    # so one grouped kernel still represents both updates exactly.
    k = a_g.shape[1]
    n = b_g.shape[0]
    if a_u.shape[1] != k or b_u.shape[0] != n:
        return False
    a_gate_up = torch.zeros((2 * r, k), device=a_g.device, dtype=a_g.dtype)
    a_gate_up[:rg].copy_(a_g)
    a_gate_up[r:r + ru].copy_(a_u)
    b_gate_up = torch.zeros((2, n, r), device=b_g.device, dtype=b_g.dtype)
    b_gate_up[0, :, :rg].copy_(b_g * float(gate_proj.scaling[name]))
    b_gate_up[1, :, :ru].copy_(b_u * float(up_proj.scaling[name]))
    a_gate_up = a_gate_up.detach()
    b_gate_up = b_gate_up.detach()
    scale = 1.0

    gate_individual = gate_proj.forward
    up_individual = up_proj.forward
    cache = {"key": None, "up": None}
    pair_plans = {}

    def grouped_pair(x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        gate_base = gate_proj.base_layer(x).reshape(-1, gate_proj.base_layer.out_features)
        up_base = up_proj.base_layer(x).reshape(-1, up_proj.base_layer.out_features)
        bases = torch.stack((gate_base, up_base), dim=1)
        both = grouped_lora(x2, a_gate_up, b_gate_up, bases, scale)
        return (
            both[:, 0].reshape(*shape[:-1], gate_proj.base_layer.out_features),
            both[:, 1].reshape(*shape[:-1], up_proj.base_layer.out_features),
        )

    def individual_pair(x):
        return gate_individual(x), up_individual(x)

    def gate_forward(self, x, *args, **kwargs):
        rows = x.numel() // x.shape[-1]
        if (
            args or kwargs or rows >= MIN_GEMM_ROWS
            or list(self.active_adapters) != [name]
            or list(up_proj.active_adapters) != [name]
            or self.merged or up_proj.merged
            or self.disable_adapters or up_proj.disable_adapters
        ):
            cache["key"] = cache["up"] = None
            return gate_individual(x, *args, **kwargs)
        plan = pair_plans.get(rows)
        if plan is None:
            candidates = {"individual": individual_pair, "grouped": grouped_pair}
            forced = os.environ.get("LORAFORGE_GROUPED_MLP", "").strip().lower()
            timings = {}
            signature = (
                x.device.type, x.device.index, x.dtype, rows, x.shape[-1],
                gate_proj.base_layer.out_features, rg, ru,
                os.environ.get("LORAFORGE_TUNE_MODE", "eager"),
            )
            cached = _GROUPED_MLP_CHOICE_CACHE.get(signature)
            if forced not in candidates and cached not in candidates:
                for candidate_name, candidate in candidates.items():
                    try:
                        timings[candidate_name] = _cuda_median_us(lambda fn=candidate: fn(x))
                    except Exception:
                        continue
            winner = forced if forced in candidates else cached
            if winner not in candidates:
                winner = min(timings, key=timings.get) if timings else "individual"
                _GROUPED_MLP_CHOICE_CACHE[signature] = winner
            elif cached in candidates:
                timings = {"cache_hit": True}
            plan = candidates[winner]
            pair_plans[rows] = plan
            mlp_module._loraforge_grouped_choices[rows] = winner
            mlp_module._loraforge_grouped_timings_us[rows] = timings
        gate, up = plan(x)
        shape = x.shape
        cache["key"] = (x.data_ptr(), tuple(shape))
        cache["up"] = up
        return gate

    def up_forward(self, x, *args, **kwargs):
        key = (x.data_ptr(), tuple(x.shape))
        if not args and not kwargs and cache["key"] == key and cache["up"] is not None:
            result = cache["up"]
            cache["key"] = cache["up"] = None
            return result
        cache["key"] = cache["up"] = None
        return up_individual(x, *args, **kwargs)

    gate_proj._loraforge_individual_forward = gate_individual
    up_proj._loraforge_individual_forward = up_individual
    gate_proj.forward = types.MethodType(gate_forward, gate_proj)
    up_proj.forward = types.MethodType(up_forward, up_proj)
    mlp_module._gate_up_packed = (a_gate_up, b_gate_up, scale, r)
    mlp_module._loraforge_grouped_choices = {}
    mlp_module._loraforge_grouped_timings_us = {}
    mlp_module._loraforge_grouped_mlp = True
    return True


def enable_loraforge_peft(
    model, enable_grouping: bool = True, cast_adapter_dtype: bool = True
) -> int:
    """Patch supported PEFT LoRA Linear modules for inference.

    PEFT commonly promotes adapter weights to fp32 for training stability even
    when the frozen model runs in bf16/fp16. That forces casts and excludes all
    low-precision kernels. For inference, cast active A/B weights back to the
    base linear dtype by default; callers requiring bitwise PEFT fp32-sidecar
    behavior can pass ``cast_adapter_dtype=False``.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoraLinear) and not hasattr(module, "_loraforge_original_forward"):
            if cast_adapter_dtype and isinstance(module.base_layer, torch.nn.Linear):
                target_dtype = module.base_layer.weight.dtype
                if target_dtype in (torch.float16, torch.bfloat16):
                    for name in module.active_adapters:
                        if name in module.lora_A and name in module.lora_B:
                            module.lora_A[name].to(dtype=target_dtype)
                            module.lora_B[name].to(dtype=target_dtype)
            module._loraforge_original_forward = module.forward
            module._loraforge_plan = None
            module._loraforge_plans = {}
            module.forward = types.MethodType(_loraforge_linear_forward, module)
            count += 1

    grouped = 0
    if enable_grouping:
        for module in model.modules():
            if _patch_grouped_mlp(module):
                grouped += 1
    model._loraforge_grouped_mlp_count = grouped

    return count


def peft_plan_report(model) -> dict:
    """Summarize selected decode implementations for diagnostics/benchmarks."""
    choices = Counter()
    timing_samples = []
    grouped_choices = Counter()
    grouped_timing_samples = []
    for module in model.modules():
        # Avoid PEFT wrapper ``__getattr__`` delegation duplicating reports.
        choice = module.__dict__.get("_loraforge_decode_choice")
        if choice is not None:
            choices[choice] += 1
            timing_samples.append({
                "choice": choice,
                "timings_us": module.__dict__.get("_loraforge_decode_timings_us", {}),
            })
        for rows, grouped_choice in module.__dict__.get("_loraforge_grouped_choices", {}).items():
            grouped_choices[grouped_choice] += 1
            grouped_timing_samples.append({
                "rows": rows,
                "choice": grouped_choice,
                "timings_us": module.__dict__.get("_loraforge_grouped_timings_us", {}).get(rows, {}),
            })
    return {
        "decode_choices": dict(choices),
        "grouped_mlp_modules": getattr(model, "_loraforge_grouped_mlp_count", 0),
        "decode_timing_samples": timing_samples,
        "grouped_mlp_choices": dict(grouped_choices),
        "grouped_mlp_timing_samples": grouped_timing_samples,
    }
