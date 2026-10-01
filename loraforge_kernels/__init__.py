from .fused_linear import (
    FUSED_MAX_RANK,
    MIN_GEMM_ROWS,
    FusedLoRALinear,
    augment_bias,
    augment_weight,
    build_ext,
    build_plan,
    kconcat_weight,
    expand_add,
    fused_gemm,
    fused_lora_linear,
    pad_rank,
    selection_report,
)
from .fused_lora import classed_lora, classed_tiled_lora, fused_lora, grouped_lora, tiled_lora
from .runtime_lora import attach_fused_lora, fused_lora_module_count, total_fused_rank
from .splitk_lora import (
    DEPLOY_RANK_BUCKETS,
    align_to_deploy_rank,
    persistent_expand_add,
    splitk_lora_shrink,
    splitk_persistent_fused_lora,
)
from .overlapped_linear import DualStreamOverlappedLinear

__all__ = [
    "fused_lora",
    "grouped_lora",
    "tiled_lora",
    "classed_lora",
    "classed_tiled_lora",
    "fused_lora_linear",
    "fused_gemm",
    "FusedLoRALinear",
    "expand_add",
    "augment_weight",
    "augment_bias",
    "kconcat_weight",
    "build_ext",
    "build_plan",
    "selection_report",
    "pad_rank",
    "FUSED_MAX_RANK",
    "MIN_GEMM_ROWS",
    "attach_fused_lora",
    "fused_lora_module_count",
    "total_fused_rank",
    "enable_loraforge_peft",
    "peft_plan_report",
    "splitk_lora_shrink",
    "persistent_expand_add",
    "splitk_persistent_fused_lora",
    "align_to_deploy_rank",
    "DEPLOY_RANK_BUCKETS",
    "DualStreamOverlappedLinear",
]


def __getattr__(name):
    """Resolve the PEFT bridge lazily.

    ``peft`` is a heavy optional dependency, and the diffusion pipelines that
    want the runtime sidecar (AnimateDiff pins diffusers 0.11 with no peft at
    all) must not fail to import this package because of it.
    """
    if name in ("enable_loraforge_peft", "peft_plan_report"):
        from . import peft_integration

        return getattr(peft_integration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
