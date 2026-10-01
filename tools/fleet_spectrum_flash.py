"""FlashTSQR spectra for every module of every adapter in a fleet manifest.

All adapters' A/B pairs are bucketed by shape and factored together with the
custom ``tsqr_factor`` kernel; singular values of ``B @ A`` come from the small
``R_B R_A^T`` core. Outputs per-module spectra and a simple fleet-wide spectral
allocation (keep the directions with the largest per-adapter-normalised energy)
at several retained-direction budgets, with compact BF16 byte counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_lora_fraq_spectrum import flash_spectra, load_flash_extension  # noqa: E402

SUFFIXES = (('.lora_A.weight', '.lora_B.weight'), ('.lora_down.weight', '.lora_up.weight'))


def adapter_pairs(path):
    state = load_file(path, device='cpu')
    pairs = []
    for key, a in state.items():
        for a_suffix, b_suffix in SUFFIXES:
            if key.endswith(a_suffix):
                pairs.append((key[:-len(a_suffix)], a, state[key[:-len(a_suffix)] + b_suffix]))
    return sorted(pairs, key=lambda p: p[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--budgets', type=float, nargs='+', default=[0.25, 0.5, 0.75])
    parser.add_argument('--flash-kernel', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'FlashTSQR/kernels/tsqr_full.cu')
    parser.add_argument('--rows-per-leaf', type=int, default=128)
    parser.add_argument('--threads-per-block', type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Refusing to overwrite a prior run')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.fleet_dir / 'manifest.json').read_text())
    device = torch.device('cuda')

    started = time.perf_counter()
    extension = load_flash_extension(args.flash_kernel)
    compile_s = time.perf_counter() - started
    print(f'FlashTSQR extension built from {args.flash_kernel} in {compile_s:.1f}s; '
          f'device={torch.cuda.get_device_name()}', flush=True)

    modules, spectra, timings = [], [], []
    for entry in manifest['entries']:
        pairs = adapter_pairs(args.fleet_dir / entry['path'])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        values = flash_spectra(pairs, extension, device, args.rows_per_leaf, args.threads_per_block)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
        for (name, a, b), s in zip(pairs, values):
            modules.append(dict(adapter=entry['name'], module=name, rank=a.shape[0],
                                d_in=a.shape[1], d_out=b.shape[0]))
            spectra.append(s.numpy())
        print(f"{entry['name']}: {len(pairs)} modules in {timings[-1]:.2f}s", flush=True)

    sigma = np.stack(spectra)  # [modules, rank]; every adapter here is uniform rank 32
    energy = sigma.astype(np.float64)**2
    adapters = [m['adapter'] for m in modules]
    names = list(dict.fromkeys(adapters))
    index = np.array([names.index(a) for a in adapters])
    totals = np.bincount(index, weights=energy.sum(1))
    normalised = energy / totals[index][:, None]
    cost = np.array([m['d_in'] + m['d_out'] for m in modules])  # params per direction

    order = np.argsort(-normalised.ravel(), kind='stable')
    full_dirs, full_params = energy.size, int(cost.sum() * sigma.shape[1])
    allocations = []
    for budget in args.budgets:
        keep = np.zeros(energy.size, bool)
        keep[order[:int(round(budget * energy.size))]] = True
        keep = keep.reshape(energy.shape)
        kept_rank = keep.sum(1)
        per_adapter = []
        for i, name in enumerate(names):
            rows = index == i
            per_adapter.append(dict(
                adapter=name, retained_energy=float((energy[rows] * keep[rows]).sum() / totals[i]),
                retained_directions=int(kept_rank[rows].sum()), modules_dropped=int((kept_rank[rows] == 0).sum()),
                compact_bf16_bytes=int(2 * (cost[rows] * kept_rank[rows]).sum())))
        params = int((cost * kept_rank).sum())
        retained = [p['retained_energy'] for p in per_adapter]
        allocations.append(dict(
            budget=budget, retained_directions=int(kept_rank.sum()), total_directions=full_dirs,
            compact_lora_parameters=params, full_lora_parameters=full_params,
            compact_bf16_bytes=2 * params, full_bf16_bytes=2 * full_params,
            min_adapter_energy=min(retained), median_adapter_energy=float(np.median(retained)),
            worst_adapter=names[int(np.argmin(retained))], per_adapter=per_adapter))
        print(f'b{int(budget*100)}: dirs {kept_rank.sum()}/{full_dirs}, '
              f'BF16 {2*full_params/1e9:.2f}->{2*params/1e9:.2f} GB, adapter energy '
              f'min {min(retained):.4f} median {np.median(retained):.4f}', flush=True)

    np.savez_compressed(args.output_dir / 'spectra.npz', sigma=sigma.astype(np.float32),
                        adapter=np.array(adapters), module=np.array([m['module'] for m in modules]))
    (args.output_dir / 'summary.json').write_text(json.dumps(dict(
        backend='FlashTSQR tsqr_factor (custom CUDA) + torch.linalg.svdvals on rank x rank core',
        kernel=str(args.flash_kernel), rows_per_leaf=args.rows_per_leaf,
        threads_per_block=args.threads_per_block, device=torch.cuda.get_device_name(),
        torch=torch.__version__, fleet=manifest['fleet'], fleet_dir=str(args.fleet_dir),
        adapters=len(names), modules=len(modules), extension_build_s=compile_s,
        spectrum_seconds_total=sum(timings),
        allocation_rule='fleet-wide top-k of sigma^2 normalised by each adapter total energy',
        allocations=allocations), indent=1) + '\n')
    print(f'wrote {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
