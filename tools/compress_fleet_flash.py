"""Compress fleet adapters to compact ranks with the FlashTSQR GPU kernel.

Ranks come from one fleet-wide allocation over a completed FlashTSQR spectrum run
(``fleet_spectrum_flash.py``): keep the directions with the largest
per-adapter-normalised energy. Each selected module is then truncated by
``vla_fleet/compress_oft_gpu.truncate`` (tsqr_factor/tsqr_applyQ around the core
SVD) and stored compactly: A is [k, d_in], B is [d_out, k], with no zero padding.
Keys keep the source naming and no alpha tensors are written, so ComfyUI/diffusers
apply scale 1 exactly as for the uncompressed (alpha-free) source files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vla_fleet'))
from compress_oft_gpu import build, truncate  # noqa: E402


def allocate(sigma, adapters, budget):
    energy = sigma.astype(np.float64)**2
    names = list(dict.fromkeys(adapters))
    index = np.array([names.index(a) for a in adapters])
    totals = np.bincount(index, weights=energy.sum(1))
    normalised = (energy / totals[index][:, None]).ravel()
    keep = np.zeros(normalised.size, bool)
    keep[np.argsort(-normalised, kind='stable')[:int(round(budget * normalised.size))]] = True
    return keep.reshape(energy.shape).sum(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet-dir', type=Path, required=True)
    parser.add_argument('--spectrum-run', type=Path, required=True)
    parser.add_argument('--adapters', nargs='+', required=True)
    parser.add_argument('--budgets', type=float, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Refusing to overwrite a prior compression run')
    manifest = json.loads((args.fleet_dir / 'manifest.json').read_text())
    entries = {e['name']: e for e in manifest['entries']}
    spectra = np.load(args.spectrum_run / 'spectra.npz')
    sigma, adapters, modules = spectra['sigma'], list(spectra['adapter']), list(spectra['module'])

    ext = build()
    print('compression backend: FlashTSQR/kernels/tsqr_full.cu', flush=True)
    records = []
    for budget in args.budgets:
        ranks = allocate(sigma, adapters, budget)
        for name in args.adapters:
            entry = entries[name]
            source = load_file(args.fleet_dir / entry['path'])
            rows = [i for i, a in enumerate(adapters) if a == name]
            out, kept, energy_kept, energy_all, worst = {}, {}, 0.0, 0.0, 0.0
            t0 = time.perf_counter()
            for i in rows:
                prefix, keep = modules[i], int(ranks[i])
                A, B = source[prefix + '.lora_A.weight'], source[prefix + '.lora_B.weight']
                An, Bn, S = truncate(ext, A, B, keep)
                # The kernel's singular values must match the spectrum run that set the ranks.
                worst = max(worst, float((S.cpu() - torch.from_numpy(sigma[i])).abs().max() / S[0].cpu()))
                energy_all += float((S**2).sum())
                energy_kept += float((S[:keep]**2).sum())
                kept[prefix] = keep
                if keep:
                    out[prefix + '.lora_A.weight'] = An[:keep].to(A.dtype).cpu().contiguous()
                    out[prefix + '.lora_B.weight'] = Bn[:, :keep].to(B.dtype).cpu().contiguous()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if worst > 1e-3:
                raise RuntimeError(f'{name}: singular values disagree with spectrum run ({worst:.2e})')
            if not all(torch.isfinite(t).all() for t in out.values()):
                raise RuntimeError(f'{name}: non-finite compressed factors')
            target = args.output_dir / f'b{round(budget*100)}' / name
            target.mkdir(parents=True, exist_ok=True)
            save_file(out, str(target / Path(entry['filename']).name), metadata={'format': 'pt'})
            stored = sum(t.numel() * t.element_size() for t in out.values())
            record = dict(
                adapter=name, budget=budget, backend='flash_tsqr',
                kernel=str(ROOT / 'FlashTSQR/kernels/tsqr_full.cu'), source=str(args.fleet_dir / entry['path']),
                source_sha256=entry['sha256'], output=str(target / Path(entry['filename']).name),
                allocation='fleet-wide top-k of per-adapter-normalised sigma^2 over all '
                           f'{len(set(adapters))} adapters', spectrum_run=str(args.spectrum_run),
                retained_directions=sum(kept.values()), total_directions=sigma.shape[1] * len(rows),
                energy_retained=energy_kept / energy_all, modules_at_zero=sum(v == 0 for v in kept.values()),
                rank_min=min(kept.values()), rank_max=max(kept.values()),
                stored_tensor_bytes=stored, source_tensor_bytes=entry['lora_parameters'] * 2,
                zero_padded=False, max_sigma_rel_diff_vs_spectrum=worst, compression_s=elapsed, ranks=kept)
            (target / 'compression.json').write_text(json.dumps(record, indent=1) + '\n')
            records.append({k: v for k, v in record.items() if k != 'ranks'})
            print(f"{name} b{round(budget*100)}: {record['retained_directions']}/{record['total_directions']} dirs, "
                  f"energy {record['energy_retained']:.4f}, {stored/1e6:.0f}/{record['source_tensor_bytes']/1e6:.0f} MB, "
                  f"ranks {record['rank_min']}-{record['rank_max']}, {elapsed:.1f}s", flush=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(records, indent=1) + '\n')


if __name__ == '__main__':
    main()
