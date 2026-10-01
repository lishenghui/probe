#!/usr/bin/env python3
"""Unified benchmark for five mathematically equivalent LoRA recompressors.

All methods compute the same best rank-k approximation to D = B @ A and return
balanced factors B_hat, A_hat with singular values split evenly between them.
The benchmark compares reconstructed updates (not basis-dependent factors).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("svd", "florist", "spectral", "fraq", "flashtsqr")


@dataclass
class Layer:
    repo_order: int
    repo_id: str
    filename: str
    module: str
    a: torch.Tensor
    b: torch.Tensor
    retained_rank: int

    @property
    def shape(self):
        return (self.b.shape[0], self.a.shape[1], self.a.shape[0])


def matching_up_key(key: str):
    suffixes = ((".lora_A.weight", ".lora_B.weight"),
                (".lora_down.weight", ".lora_up.weight"),
                (".lora.down.weight", ".lora.up.weight"))
    for down, up in suffixes:
        if key.endswith(down):
            return key[:-len(down)], key[:-len(down)] + up
    match = re.match(r"^(.*)\.lora_A\.([^.]+)\.weight$", key)
    if match:
        prefix, adapter = match.groups()
        return prefix, f"{prefix}.lora_B.{adapter}.weight"
    return None


def matrix_pair(a: torch.Tensor, b: torch.Tensor):
    if a.ndim < 2 or b.ndim < 2 or b.shape[1] != a.shape[0]:
        return None
    if b.ndim > 2 and math.prod(b.shape[2:]) != 1:
        return None
    return a.reshape(a.shape[0], -1), b.reshape(b.shape[0], a.shape[0])


def locate_file(root: Path, repo_id: str, filename: str) -> Path | None:
    direct = root / repo_id / filename
    if direct.is_file():
        return direct
    hits = list((root / repo_id).rglob(Path(filename).name)) if (root / repo_id).is_dir() else []
    suffix_hits = [p for p in hits if p.as_posix().endswith(filename)]
    if len(suffix_hits) == 1:
        return suffix_hits[0]
    return hits[0] if len(hits) == 1 else None


def load_top_layers(adapter_root: Path, layers_csv: Path, projects: int,
                    layers_per_project: int) -> tuple[list[Layer], list[dict]]:
    by_repo = defaultdict(list)
    repo_order = {}
    with layers_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["repo_id"] not in repo_order:
                repo_order[row["repo_id"]] = len(repo_order) + 1
            order = repo_order[row["repo_id"]]
            if order <= projects:
                by_repo[order].append(row)
    layers, coverage = [], []
    for order in range(1, projects + 1):
        rows = by_repo[order]
        repo_id = rows[0]["repo_id"] if rows else "missing"
        selected, seen = [], set()
        # Spread samples over files/modules rather than taking adjacent layers only.
        stride = max(1, len(rows) // max(1, layers_per_project))
        candidates = rows[::stride] + rows
        for row in candidates:
            identity = (row["filename"], row.get("module", ""))
            if identity in seen:
                continue
            seen.add(identity)
            path = locate_file(adapter_root, repo_id, row["filename"])
            if path is None:
                continue
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                # Census CSV module is the exact prefix. Resolve its A key robustly.
                options = [f"{row['module']}{suffix}" for suffix in
                           (".lora_A.weight", ".lora_down.weight", ".lora.down.weight")]
                options += [k for k in keys if k.startswith(row["module"] + ".lora_A.") and k.endswith(".weight")]
                a_key = next((k for k in options if k in keys), None)
                match = matching_up_key(a_key) if a_key else None
                if not match or match[1] not in keys:
                    continue
                pair = matrix_pair(handle.get_tensor(a_key), handle.get_tensor(match[1]))
            if pair is None:
                continue
            a, b = pair
            selected.append(Layer(order, repo_id, row["filename"], row["module"], a, b,
                                  int(row["r95"])))
            if len(selected) >= layers_per_project:
                break
        layers.extend(selected)
        coverage.append({"repo_order": order, "repo_id": repo_id,
                         "available_layers": len(rows), "sampled_layers": len(selected)})
    return layers, coverage


def balanced(u, s, vh, k):
    root = s[..., :k].clamp_min(0).sqrt()
    return u[..., :, :k] * root.unsqueeze(-2), root.unsqueeze(-1) * vh[..., :k, :]


def dense_svd(a, b, k):
    u, s, vh = torch.linalg.svd(b @ a, full_matrices=False)
    return (*balanced(u, s, vh, k), s)


def florist(a, b, k):
    # Double-sided SVD: B=Ub Sb VbH, A=Ua Sa VaH; only SVD the r x r core.
    ub, sb, vbh = torch.linalg.svd(b, full_matrices=False)
    ua, sa, vah = torch.linalg.svd(a, full_matrices=False)
    core = (sb.unsqueeze(-1) * vbh) @ ua @ (sa.unsqueeze(-1) * torch.eye(sa.shape[-1], device=a.device))
    uc, s, vhc = torch.linalg.svd(core, full_matrices=False)
    left, right = balanced(uc, s, vhc, k)
    return ub @ left, right @ vah, s


def spectral(a, b, k):
    # SpecTraL: double QR followed by an SVD of the r x r core.
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.mT.contiguous(), mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.mT, full_matrices=False)
    left, right = balanced(u, s, vh, k)
    return qb @ left, right @ qa.mT, s


def fraq(a, b, k):
    # QR only the factor with the smaller ambient dimension, then EVD a Gram matrix.
    if b.shape[0] <= a.shape[1]:
        q, r = torch.linalg.qr(b, mode="reduced")
        h = r @ a
        vals, vecs = torch.linalg.eigh(h @ h.mT)
        vals, vecs = vals.flip(-1), vecs.flip(-1).contiguous()
        s = vals.clamp_min(0).sqrt()
        root = s[:k].sqrt()
        inv = torch.where(root > 1e-12, root.reciprocal(), torch.zeros_like(root))
        bh = (q @ vecs[:, :k]) * root
        ah = (vecs[:, :k].mT @ h) * inv[:, None]
    else:
        q, r = torch.linalg.qr(a.mT.contiguous(), mode="reduced")
        h = b @ r.mT
        vals, vecs = torch.linalg.eigh(h.mT @ h)
        vals, vecs = vals.flip(-1), vecs.flip(-1).contiguous()
        s = vals.clamp_min(0).sqrt()
        root = s[:k].sqrt()
        inv = torch.where(root > 1e-12, root.reciprocal(), torch.zeros_like(root))
        ah = (q @ vecs[:, :k] * root).mT
        bh = (h @ vecs[:, :k]) * inv
    return bh, ah, s


def flash_fraq(a, b, k, ext, rows_per_leaf, tpb):
    # Same one-sided FraQ algebra; custom batched TSQR supplies R and Q@V.
    # Extension state is overwritten by each factor call, hence factor/apply stay adjacent.
    if b.shape[0] <= a.shape[1]:
        x, other, left = b, a, True
    else:
        x, other, left = a.mT.contiguous(), b.mT.contiguous(), False
    xb = x.unsqueeze(0).contiguous()
    rpl = max(x.shape[1], min(rows_per_leaf, x.shape[0]))
    r = ext.tsqr_factor(xb, rpl, tpb)[0]
    h = r @ other if left else r @ other
    vals, vecs = torch.linalg.eigh(h @ h.mT)
    vals, vecs = vals.flip(-1), vecs.flip(-1).contiguous()
    s = vals.clamp_min(0).sqrt()
    root = s[:k].sqrt()
    inv = torch.where(root > 1e-12, root.reciprocal(), torch.zeros_like(root))
    qv = ext.tsqr_applyQ(vecs[:, :k].unsqueeze(0).contiguous(), tpb)[0]
    if left:
        return qv * root, (vecs[:, :k].mT @ h) * inv[:, None], s
    # Here A^T=Q R and h=R B^T, so transpose the reconstructed h-side factor.
    return ((vecs[:, :k].mT @ h) * inv[:, None]).mT, (qv * root).mT, s


def batched_fraq(a, b, ks):
    """True batched one-sided FraQ for one homogeneous shape bucket."""
    left = b.shape[1] <= a.shape[2]
    if left:
        q, r = torch.linalg.qr(b, mode="reduced")
        h = r @ a
    else:
        q, r = torch.linalg.qr(a.transpose(1, 2).contiguous(), mode="reduced")
        h = r @ b.transpose(1, 2)
    vals, vecs = torch.linalg.eigh(h @ h.transpose(1, 2))
    vals, vecs = vals.flip(-1), vecs.flip(-1).contiguous()
    s = vals.clamp_min(0).sqrt()
    max_k = max(ks)
    root = s[:, :max_k].sqrt()
    inv = torch.where(root > 1e-12, root.reciprocal(), torch.zeros_like(root))
    qv = (q @ vecs[:, :, :max_k]) * root[:, None, :]
    other = (vecs[:, :, :max_k].transpose(1, 2) @ h) * inv[:, :, None]
    return (qv, other, s, left)


def batched_flash_fraq(a, b, ks, ext, rows_per_leaf, tpb):
    """Same batched FraQ algebra with FlashTSQR factor/apply kernels."""
    left = b.shape[1] <= a.shape[2]
    x = b if left else a.transpose(1, 2).contiguous()
    other_input = a if left else b.transpose(1, 2).contiguous()
    rpl = max(x.shape[2], min(rows_per_leaf, x.shape[1]))
    r = ext.tsqr_factor(x, rpl, tpb)
    h = r @ other_input
    vals, vecs = torch.linalg.eigh(h @ h.transpose(1, 2))
    vals, vecs = vals.flip(-1), vecs.flip(-1).contiguous()
    s = vals.clamp_min(0).sqrt()
    max_k = max(ks)
    root = s[:, :max_k].sqrt()
    inv = torch.where(root > 1e-12, root.reciprocal(), torch.zeros_like(root))
    qv = ext.tsqr_applyQ(vecs[:, :, :max_k].contiguous(), tpb) * root[:, None, :]
    other = (vecs[:, :, :max_k].transpose(1, 2) @ h) * inv[:, :, None]
    return (qv, other, s, left)


def rel_product_error(b1, a1, b2, a2):
    # ||B1A1-B2A2||_F using only small Gram/cross matrices.
    n1 = torch.trace((b1.mT @ b1) @ (a1 @ a1.mT))
    n2 = torch.trace((b2.mT @ b2) @ (a2 @ a2.mT))
    cross = torch.trace((b1.mT @ b2) @ (a2 @ a1.mT))
    return ((n1 + n2 - 2 * cross).clamp_min(0) / n1.clamp_min(1e-30)).sqrt().item()


def energy_rank(s, threshold):
    energy = s.square()
    return min(len(s), int(torch.searchsorted(energy.cumsum(0) / energy.sum(), threshold).item()) + 1)


def fingerprints(b, a, count=32):
    """Deterministic entries of B@A for cross-job output equivalence checks."""
    m, n = b.shape[0], a.shape[1]
    values = []
    for t in range(count):
        i = (t * 104729 + 17) % m
        j = (t * 130363 + 29) % n
        values.append(float(torch.dot(b[i].float(), a[:, j].float()).item()))
    return values


def timed(fn, device, warmup, reps):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        values = []
        for _ in range(reps):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(); out = fn(); end.record(); torch.cuda.synchronize()
            values.append(start.elapsed_time(end))
    else:
        values = []
        for _ in range(reps):
            start = time.perf_counter(); out = fn(); values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-root", type=Path, default=ROOT / "artifacts/hf_lora_census/top100_adapters")
    p.add_argument("--layers-csv", type=Path, default=ROOT / "artifacts/hf_lora_census/top100_rank32_64/layers.csv")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--projects", type=int, default=10)
    p.add_argument("--layers-per-project", type=int, default=10)
    p.add_argument("--energy", type=float, default=.95)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--rows-per-leaf", type=int, default=128)
    p.add_argument("--threads-per-block", type=int, default=256)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--true-batch", action="store_true",
                   help="bucket homogeneous layers and run FraQ/FlashTSQR as real batches")
    p.add_argument("--flash-kernel", type=Path, default=ROOT / "FlashTSQR/kernels/tsqr_full.cu")
    args = p.parse_args()
    if not 0 < args.energy <= 1:
        p.error("--energy must be in (0,1]")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA allocation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers, coverage = load_top_layers(args.adapter_root, args.layers_csv,
                                       args.projects, args.layers_per_project)
    sampled_layer_count = len(layers)
    if len({x.repo_order for x in layers}) != args.projects:
        raise RuntimeError(f"Only covered {len({x.repo_order for x in layers})}/{args.projects} repos: {coverage}")

    from analyze_lora_fraq_spectrum import load_flash_extension
    ext = load_flash_extension(args.flash_kernel)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    print(f"GPU: {torch.cuda.get_device_name(0)}; method={args.method}; sampled {len(layers)} layers from {args.projects} repos", flush=True)
    if args.true_batch:
        if args.method not in ("fraq", "flashtsqr"):
            raise ValueError("--true-batch is supported for fraq/flashtsqr")
        buckets = defaultdict(list)
        for layer in layers:
            out_dim, in_dim, rank = layer.shape
            left = out_dim <= in_dim
            qr_rows = out_dim if left else in_dim
            other_dim = in_dim if left else out_dim
            buckets[(layer.repo_order, qr_rows, other_dim, rank, left)].append(layer)
        print("batch histogram:", sorted((len(v) for v in buckets.values()), reverse=True), flush=True)
        for bucket_index, group in enumerate(buckets.values(), 1):
            a = torch.stack([x.a for x in group]).to(device=device, dtype=torch.float32)
            b = torch.stack([x.b for x in group]).to(device=device, dtype=torch.float32)
            ks = [min(x.retained_rank, x.a.shape[0]) for x in group]
            fn = ((lambda: batched_fraq(a, b, ks)) if args.method == "fraq" else
                  (lambda: batched_flash_fraq(a, b, ks, ext, args.rows_per_leaf, args.threads_per_block)))
            batch_ms, (qv, other, spectra, left) = timed(fn, device, args.warmup, args.reps)
            for pos, (layer, k) in enumerate(zip(group, ks)):
                if left:
                    bh, ah = qv[pos, :, :k], other[pos, :k, :]
                else:
                    bh, ah = other[pos, :k, :].mT, qv[pos, :, :k].mT
                residual = rel_product_error(b[pos], a[pos], bh, ah)
                rows.append({"repo_order": layer.repo_order, "repo_id": layer.repo_id,
                             "filename": layer.filename, "module": layer.module,
                             "out_dim": b.shape[1], "in_dim": a.shape[2], "rank": a.shape[1],
                             "retained_rank": k, "energy_target": args.energy, "method": args.method,
                             "median_ms": batch_ms / len(group), "batch_ms": batch_ms,
                             "batch_size": len(group), "rel_reconstruction_residual": residual,
                             "spectrum": json.dumps([float(x) for x in spectra[pos].cpu()]),
                             "fingerprint": json.dumps(fingerprints(bh, ah))})
            print(f"[{bucket_index}/{len(buckets)}] batch={len(group)} shape={group[0].shape} "
                  f"batch_ms={batch_ms:.4f} per_layer_ms={batch_ms/len(group):.4f}", flush=True)
        # Skip the legacy per-layer loop below.
        layers = []
    for index, layer in enumerate(layers, 1):
        a = layer.a.to(device=device, dtype=torch.float32)
        b = layer.b.to(device=device, dtype=torch.float32)
        k = min(layer.retained_rank, a.shape[0])
        fns = {
            "svd": lambda: dense_svd(a, b, k),
            "florist": lambda: florist(a, b, k),
            "spectral": lambda: spectral(a, b, k),
            "fraq": lambda: fraq(a, b, k),
            "flashtsqr": lambda: flash_fraq(a, b, k, ext, args.rows_per_leaf, args.threads_per_block),
        }
        warmup = 0 if args.method == "svd" else args.warmup
        reps = 1 if args.method == "svd" else args.reps
        ms, (bh, ah, spectrum) = timed(fns[args.method], device, warmup, reps)
        residual = rel_product_error(b, a, bh, ah)
        rows.append({"repo_order": layer.repo_order, "repo_id": layer.repo_id,
                     "filename": layer.filename, "module": layer.module,
                     "out_dim": b.shape[0], "in_dim": a.shape[1], "rank": a.shape[0],
                     "retained_rank": k, "energy_target": args.energy, "method": args.method,
                     "median_ms": ms, "batch_ms": ms, "batch_size": 1,
                     "rel_reconstruction_residual": residual,
                     "spectrum": json.dumps([float(x) for x in spectrum[:a.shape[0]].cpu()]),
                     "fingerprint": json.dumps(fingerprints(bh, ah))})
        print(f"[{index}/{len(layers)}] repo={layer.repo_order} shape={layer.shape} k={k}", flush=True)
        del a, b, bh, ah, spectrum
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "method": args.method,
               "projects": args.projects, "sampled_layers": sampled_layer_count, "energy": args.energy,
               "coverage": coverage, "methods": {}}
    for method in (args.method,):
        selected = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {
            "median_layer_ms": statistics.median(r["median_ms"] for r in selected),
            "total_median_ms": sum(r["median_ms"] for r in selected),
            "max_reconstruction_residual": max(r["rel_reconstruction_residual"] for r in selected),
            "median_reconstruction_residual": statistics.median(r["rel_reconstruction_residual"] for r in selected),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
