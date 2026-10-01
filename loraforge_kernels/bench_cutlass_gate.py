#!/usr/bin/env python3
"""Gate: does a plain CUTLASS SM90 GEMM match cuBLAS on our layer shapes?

Fusing the LoRA expand into a GEMM epilogue only pays if the GEMM is
cuBLAS-class to begin with.  The Triton attempt failed this gate (1.4-1.9x
slower before any sidecar work), so measure it before building anything on top.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]

SHAPES = [
    ("wan/attn-1536", 32760, 1536, 1536),
    ("wan/ffn-up-8960", 32760, 1536, 8960),
    ("wan/ffn-down-1536", 32760, 8960, 1536),
    ("llm/prefill-4096", 4096, 4096, 4096),
]


def cuda_include_paths():
    """torch's ATen headers pull in cusparse/cublas, which the repo's minimal
    nvcc package does not carry.  The torch wheel vendors them under
    site-packages/nvidia/*/include, so hand those to the compiler."""
    import torch as _torch
    nvidia = Path(_torch.__file__).resolve().parents[1] / "nvidia"
    return [str(p) for p in sorted(nvidia.glob("*/include")) if p.is_dir()]


def build():
    return load(
        name="loraforge_cutlass_gate",
        sources=[str(ROOT / "loraforge_kernels/csrc/cutlass_gemm_gate.cu")],
        extra_include_paths=[
            str(ROOT / "third_party/cutlass/include"),
            str(ROOT / "third_party/cutlass/tools/util/include"),
        ] + cuda_include_paths(),
        extra_cuda_cflags=[
            "-O3", "-std=c++17",
            "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "-gencode", "arch=compute_90a,code=sm_90a",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        ],
        verbose=True,
    )


def time_all(fns, warmup=10, rounds=30):
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    best = {k: float("inf") for k in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            b, e = torch.cuda.Event(True), torch.cuda.Event(True)
            b.record(); fn(); e.record(); e.synchronize()
            best[name] = min(best[name], b.elapsed_time(e))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kernel_bench/cutlass_gate.json")
    args = parser.parse_args()

    ext = build()
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    rows = []
    for label, m, k, n in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.5
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
        reference = torch.nn.functional.linear(x, w).float()

        fns = {"cublas": lambda: torch.nn.functional.linear(x, w)}
        errs = {}
        for cfg in (0, 1, 2):
            try:
                got = ext.cutlass_gemm(x, w, cfg)
            except Exception as exc:  # a tile shape that cannot run drops out
                print(f"  config {cfg} unavailable: {exc}")
                continue
            denom = reference.abs().max().clamp_min(1e-6)
            errs[f"cutlass{cfg}"] = float((got.float() - reference).abs().max() / denom)
            fns[f"cutlass{cfg}"] = (lambda c=cfg: ext.cutlass_gemm(x, w, c))

        t = time_all(fns)
        base = t["cublas"]
        flops = 2 * m * k * n
        print(f"\n=== {label}  M={m} K={k} N={n}")
        for name, ms in sorted(t.items(), key=lambda kv: kv[1]):
            tag = "" if name == "cublas" else f"  {base/ms:5.2f}x vs cuBLAS  rel_err={errs.get(name, 0):.2e}"
            print(f"  {name:10s} {ms*1000:8.1f} us  {flops/ms*1e-9:7.1f} TFLOP/s{tag}")
            rows.append({"shape": label, "m": m, "k": k, "n": n, "impl": name,
                         "us": ms * 1000, "tflops": flops / ms * 1e-9,
                         "ratio_vs_cublas": base / ms, "rel_err": errs.get(name, 0.0)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.output}")

    print("\n=== verdict: best CUTLASS config vs cuBLAS, per shape ===")
    for label, *_ in SHAPES:
        hits = [r for r in rows if r["shape"] == label and r["impl"] != "cublas"]
        if hits:
            best = max(hits, key=lambda r: r["ratio_vs_cublas"])
            print(f"  {label:20s} {best['impl']:10s} {best['ratio_vs_cublas']:5.2f}x")


if __name__ == "__main__":
    main()
