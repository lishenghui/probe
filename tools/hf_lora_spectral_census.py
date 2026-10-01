#!/usr/bin/env python3
"""Download popular Hugging Face LoRAs and measure their update spectra.

The singular values of B @ A are computed from the small core obtained after
thin QR factorizations.  With ``--backend flash`` the R factors are produced by
the batched FlashTSQR kernel in this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_lora_fraq_spectrum import load_flash_extension  # noqa: E402


ENERGY_LEVELS = (0.90, 0.95, 0.98, 0.99)
WEIGHT_SUFFIXES = (
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
    (".lora.down.weight", ".lora.up.weight"),
)


@dataclass
class Pair:
    module: str
    a: torch.Tensor
    b: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])


def jsonable_model(model) -> dict:
    siblings = [getattr(x, "rfilename", str(x)) for x in (model.siblings or [])]
    return {
        "repo_id": model.id,
        "downloads": int(model.downloads or 0),
        "downloads_all_time": getattr(model, "downloads_all_time", None),
        "created_at": str(getattr(model, "created_at", "") or ""),
        "last_modified": str(getattr(model, "last_modified", "") or ""),
        "tags": list(model.tags or []),
        "siblings": siblings,
    }


def discover(args) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    expand = ["downloads", "downloadsAllTime", "createdAt", "lastModified", "tags", "siblings"]
    pools = []
    # PEFT is the reliable pool. The lora search adds Diffusers/Kohya repos that
    # do not carry the PEFT library tag; all hits are validated from tensors.
    queries = [("peft", None)]
    if args.include_search_lora:
        queries.append((None, "lora"))
    for model_filter, search in queries:
        kwargs = dict(sort="downloads", limit=args.candidate_limit, expand=expand)
        if model_filter:
            kwargs["filter"] = model_filter
        if search:
            kwargs["search"] = search
        pools.extend(api.list_models(**kwargs))
    unique = {m.id: m for m in pools}
    models = sorted(unique.values(), key=lambda m: int(m.downloads or 0), reverse=True)
    records = [jsonable_model(m) for m in models]
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Discovered {len(records)} unique candidates -> {args.metadata}", flush=True)
    return records


def candidate_weight_files(record: dict) -> list[str]:
    files = record.get("siblings", [])
    safe = [f for f in files if f.endswith(".safetensors")]
    preferred = [
        f for f in safe
        if Path(f).name in {"adapter_model.safetensors", "pytorch_lora_weights.safetensors"}
        or re.search(r"(?:^|[/_.-])lora(?:[/_.-]|$)", f, re.I)
    ]
    # Avoid accidentally downloading full-model checkpoints from mixed repos.
    return sorted(preferred, key=lambda f: (Path(f).name != "adapter_model.safetensors", f))


def download_file(repo_id: str, filename: str, args) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id, filename, cache_dir=str(args.cache_dir), token=args.token,
        local_files_only=args.local_files_only,
    ))


def tensor_to_matrix(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if a.ndim < 2 or b.ndim < 2:
        return None
    rank = a.shape[0]
    if b.shape[1] != rank:
        return None
    # Exact matrix interpretation for Linear and the usual Conv LoRA with a
    # spatial down projection followed by a 1x1 up projection.
    if b.ndim > 2 and math.prod(b.shape[2:]) != 1:
        return None
    return a.reshape(rank, -1), b.reshape(b.shape[0], rank)


def matching_up_key(key: str) -> tuple[str, str] | None:
    """Return (module prefix, up/B key), including named PEFT adapters."""
    for down, up in WEIGHT_SUFFIXES:
        if key.endswith(down):
            return key[:-len(down)], key[:-len(down)] + up
    match = re.match(r"^(.*)\.lora_A\.([^.]+)\.weight$", key)
    if match:
        prefix, adapter = match.groups()
        return prefix, f"{prefix}.lora_B.{adapter}.weight"
    return None


def load_pairs(path: Path, ranks: set[int]) -> tuple[list[Pair], dict]:
    from safetensors import safe_open

    pairs, skipped = [], Counter()
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            match = matching_up_key(key)
            if match is None:
                continue
            prefix, b_key = match
            if b_key not in keys:
                skipped["missing_pair"] += 1
                continue
            converted = tensor_to_matrix(handle.get_tensor(key), handle.get_tensor(b_key))
            if converted is None:
                skipped["unsupported_shape"] += 1
                continue
            a, b = converted
            if a.shape[0] not in ranks:
                skipped["rank_not_selected"] += 1
                continue
            pairs.append(Pair(prefix, a, b))
    return pairs, dict(skipped)


def torch_spectra(pairs: list[Pair], device: torch.device) -> list[np.ndarray]:
    output = []
    for pair in pairs:
        a = pair.a.to(device=device, dtype=torch.float32)
        b = pair.b.to(device=device, dtype=torch.float32)
        rb = torch.linalg.qr(b, mode="reduced").R
        ra = torch.linalg.qr(a.T, mode="reduced").R
        output.append(torch.linalg.svdvals(rb @ ra.T).cpu().numpy())
    return output


def flash_spectra(pairs: list[Pair], extension, args) -> tuple[list[np.ndarray], list[str]]:
    output: list[np.ndarray | None] = [None] * len(pairs)
    used = ["flash"] * len(pairs)
    buckets = defaultdict(list)
    for i, pair in enumerate(pairs):
        buckets[(tuple(pair.a.shape), tuple(pair.b.shape))].append(i)
    for indices in buckets.values():
        sample = pairs[indices[0]]
        rank = sample.rank
        # The current kernel supports N <= about 235 and requires tall matrices.
        if rank > 235 or min(sample.a.shape[1], sample.b.shape[0]) < rank:
            values = torch_spectra([pairs[i] for i in indices], torch.device(args.device))
            for i, value in zip(indices, values):
                output[i], used[i] = value, "torch_fallback"
            continue
        a = torch.stack([pairs[i].a for i in indices]).to(args.device, torch.float32)
        b = torch.stack([pairs[i].b for i in indices]).to(args.device, torch.float32)
        rb = extension.tsqr_factor(b, max(rank, min(args.rows_per_leaf, b.shape[1])), args.threads_per_block)
        ra = extension.tsqr_factor(a.transpose(1, 2).contiguous(), max(rank, min(args.rows_per_leaf, a.shape[2])), args.threads_per_block)
        values = torch.linalg.svdvals(rb @ ra.transpose(1, 2)).cpu().numpy()
        for i, value in zip(indices, values):
            output[i] = value
    return [x for x in output if x is not None], used


def metrics(s: np.ndarray) -> dict:
    energy = np.square(s.astype(np.float64))
    total = energy.sum()
    if not np.isfinite(total) or total <= 0:
        return {"r90": 0, "r95": 0, "r99": 0, "stable_rank": 0.0, "effective_rank": 0.0}
    p = energy / total
    cumulative = np.cumsum(p)
    return {
        "r90": int(np.searchsorted(cumulative, .90) + 1),
        "r95": int(np.searchsorted(cumulative, .95) + 1),
        "r99": int(np.searchsorted(cumulative, .99) + 1),
        "stable_rank": float(total / energy[0]) if energy[0] else 0.0,
        "effective_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def interpolate_curves(spectra: list[np.ndarray], cumulative: bool) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(1 / max(map(len, spectra)), 1.0, 256)
    curves = []
    for s in spectra:
        y = np.cumsum(s.astype(np.float64) ** 2)
        y = y / y[-1] if cumulative and y[-1] else y
        if not cumulative:
            y = s / s[0] if s[0] else s
        x = np.arange(1, len(s) + 1) / len(s)
        curves.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
    return grid, np.asarray(curves)


def make_plots(rows: list[dict], spectra: list[np.ndarray], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    grid, decay = interpolate_curves(spectra, False)
    _, cumulative = interpolate_curves(spectra, True)
    for values, name, ylabel, logy in (
        (decay, "spectrum_decay.png", r"$\sigma_k/\sigma_1$", True),
        (cumulative, "cumulative_energy.png", "Cumulative squared spectral energy", False),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        # A deterministic subset of faint layer curves keeps vector/raster size sane.
        for curve in values[:min(1000, len(values))]:
            ax.plot(grid, curve, color="#4C78A8", alpha=.025, linewidth=.5)
        q25, med, q75 = np.quantile(values, [.25, .5, .75], axis=0)
        ax.fill_between(grid, q25, q75, color="#F58518", alpha=.22, label="layer IQR")
        ax.plot(grid, med, color="#E45756", linewidth=2.3, label="layer median")
        if not logy:
            for level in ENERGY_LEVELS:
                ax.axhline(level, color="gray", linestyle="--", linewidth=.65)
        else:
            ax.set_yscale("log")
        ax.set(xlabel="Retained rank fraction  k/r", ylabel=ylabel, xlim=(0, 1))
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(output_dir / name, dpi=220); plt.close(fig)


def analyze(args, records: list[dict]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    extension = None
    if args.backend == "flash":
        if device.type != "cuda":
            raise ValueError("--backend flash requires --device cuda")
        extension = load_flash_extension(args.flash_kernel)

    layer_rows, repo_rows, spectra, names = [], [], [], []
    failures = []
    for record in records:
        if len(repo_rows) >= args.limit:
            break
        repo_id = record["repo_id"]
        files = candidate_weight_files(record)
        if not files:
            failures.append({"repo_id": repo_id, "reason": "no_lora_safetensors_candidate"}); continue
        accepted = []
        repo_skipped = Counter()
        for filename in files[:args.max_files_per_repo]:
            try:
                path = download_file(repo_id, filename, args)
                pairs, skipped = load_pairs(path, set(args.ranks))
                repo_skipped.update(skipped)
                accepted.extend((filename, pair) for pair in pairs)
            except Exception as exc:
                failures.append({"repo_id": repo_id, "file": filename, "reason": f"{type(exc).__name__}: {exc}"})
        if not accepted:
            failures.append({"repo_id": repo_id, "reason": "no_selected_rank_pairs", "detail": dict(repo_skipped)}); continue
        pairs = [p for _, p in accepted]
        started = time.perf_counter()
        if extension is None:
            repo_spectra = torch_spectra(pairs, device); backends = ["torch"] * len(pairs)
        else:
            repo_spectra, backends = flash_spectra(pairs, extension, args)
        elapsed = time.perf_counter() - started
        for (filename, pair), s, backend in zip(accepted, repo_spectra, backends):
            row = {
                "repo_id": repo_id, "downloads": record["downloads"], "filename": filename,
                "module": pair.module, "input_dim": pair.a.shape[1], "output_dim": pair.b.shape[0],
                "nominal_rank": pair.rank, **metrics(s), "backend": backend,
            }
            for level in (90, 95, 99): row[f"r{level}_fraction"] = row[f"r{level}"] / pair.rank
            layer_rows.append(row); spectra.append(s); names.append(f"{repo_id}::{filename}::{pair.module}")
        selected_rows = layer_rows[-len(pairs):]
        repo_rows.append({
            "popularity_order": len(repo_rows) + 1, "repo_id": repo_id, "downloads": record["downloads"],
            "files": len(set(x[0] for x in accepted)), "layers": len(pairs),
            "ranks": ";".join(map(str, sorted({p.rank for p in pairs}))),
            "median_r90_fraction": float(np.median([r["r90_fraction"] for r in selected_rows])),
            "median_r95_fraction": float(np.median([r["r95_fraction"] for r in selected_rows])),
            "median_r99_fraction": float(np.median([r["r99_fraction"] for r in selected_rows])),
            "elapsed_seconds": elapsed,
        })
        print(f"[{len(repo_rows):3d}/{args.limit}] {repo_id}: {len(pairs)} layers, median r95/r={repo_rows[-1]['median_r95_fraction']:.3f}", flush=True)

    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if not layer_rows:
        raise RuntimeError("No analyzable LoRA layers found; see failures.json")
    write_csv(args.output_dir / "layers.csv", layer_rows)
    write_csv(args.output_dir / "repos.csv", repo_rows)
    # Store a dense NaN-padded array instead of an object array so consumers do
    # not need allow_pickle=True. ``nominal_rank`` identifies the valid prefix.
    spectrum_matrix = np.full((len(spectra), max(map(len, spectra))), np.nan, dtype=np.float32)
    for i, values in enumerate(spectra):
        spectrum_matrix[i, :len(values)] = values
    np.savez_compressed(
        args.output_dir / "spectra.npz", names=np.asarray(names),
        nominal_rank=np.asarray([len(x) for x in spectra]), singular_values=spectrum_matrix,
    )
    summary = {
        "repos": len(repo_rows), "layers": len(layer_rows), "downloads_sum": sum(r["downloads"] for r in repo_rows),
        "selected_ranks": args.ranks, "backend_requested": args.backend,
        **{f"median_{key}_fraction": float(np.median([r[f"{key}_fraction"] for r in layer_rows])) for key in ("r90", "r95", "r99")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    try:
        make_plots(layer_rows, spectra, args.output_dir)
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print("matplotlib is unavailable; skipped PNG generation", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("run", "discover", "analyze"), default="run")
    p.add_argument("--limit", type=int, default=100, help="number of valid repositories")
    p.add_argument("--candidate-limit", type=int, default=1000)
    p.add_argument("--ranks", type=int, nargs="+", default=[32, 64])
    p.add_argument("--include-search-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-files-per-repo", type=int, default=2)
    p.add_argument("--metadata", type=Path, default=root / "artifacts/hf_lora_census/candidates.json")
    p.add_argument("--output-dir", type=Path, default=root / "artifacts/hf_lora_census/top100_rank32_64")
    p.add_argument("--cache-dir", type=Path, default=root / "artifacts/hf_lora_census/hf_cache")
    p.add_argument("--backend", choices=("torch", "flash"), default="flash")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flash-kernel", type=Path, default=root / "FlashTSQR/kernels/tsqr_full.cu")
    p.add_argument("--rows-per-leaf", type=int, default=128)
    p.add_argument("--threads-per-block", type=int, default=256)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--token", default=None, help="normally use the HF_TOKEN environment variable")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode in ("run", "discover"):
        records = discover(args)
        if args.mode == "discover": return
    else:
        records = json.loads(args.metadata.read_text())
    analyze(args, records)


if __name__ == "__main__":
    main()
