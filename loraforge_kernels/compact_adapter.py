"""Export genuinely compact, heterogeneous PEFT Linear LoRAs without rescaling.

The input must already be zero-padded to its allocated ranks. This is a storage
and execution transformation, not a new truncation or allocation algorithm.
"""

import argparse
import copy
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def module_name(a_key):
    suffix = '.lora_A.weight'
    if not a_key.endswith(suffix):
        raise ValueError(f'Expected a PEFT Linear LoRA key: {a_key}')
    return a_key.removesuffix(suffix).removeprefix('base_model.model.')


def pattern_value(config, field, default, name):
    # Match PEFT's ordered suffix-regex semantics, including anchored patterns.
    for pattern, value in config.get(field, {}).items():
        if re.match(rf'(.*\.)?({pattern})$', name):
            return value
    return config[default]


def compact_zero_padded(tensors, config, ranks):
    """Return compact state/config plus a per-layer proof of shape and scaling.

    Rank-zero targets disappear entirely; no rank-one dummy adapter is used.
    An all-zero allocation is represented as base-only in the manifest and is
    intentionally not presented as a loadable, empty PEFT adapter.
    """
    if (config.get('peft_type', 'LORA') != 'LORA' or config.get('use_dora')
            or config.get('lora_bias') or config.get('bias', 'none') != 'none'
            or config.get('modules_to_save') or config.get('target_parameters')
            or config.get('layer_replication') or config.get('use_qalora')
            or config.get('alora_invocation_tokens')):
        raise ValueError('Compaction currently supports ordinary bias-free Linear LoRA/rsLoRA only')
    keys = sorted(k for k in tensors if k.endswith('.lora_A.weight'))
    if not keys or len(ranks) != len(keys):
        raise ValueError(f'Allocation has {len(ranks)} ranks for {len(keys)} A/B pairs')
    output = {}
    cfg = copy.deepcopy(config)
    cfg['rank_pattern'], cfg['alpha_pattern'], cfg['target_modules'] = {}, {}, []
    cfg['layers_to_transform'] = None
    cfg['layers_pattern'] = None
    cfg['exclude_modules'] = None
    cfg['inference_mode'] = True
    details = []
    paired = set()
    generator = torch.Generator().manual_seed(91)
    for a_key, rank in zip(keys, ranks):
        b_key = a_key.replace('.lora_A.weight', '.lora_B.weight')
        a, b = tensors[a_key], tensors[b_key]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f'Unsupported factor shapes for {a_key}')
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank <= a.shape[0]:
            raise ValueError(f'Invalid retained rank {rank} for {a_key}')
        # Removing a coordinate is exact if at least one of its factors is zero.
        removed_live = a[rank:].ne(0).any(1) & b[:, rank:].ne(0).any(0)
        if bool(removed_live.any()):
            raise ValueError(f'{a_key}: discarded coordinates are nonzero; truncate before compacting')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f'Nonfinite factors: {a_key}')
        name = module_name(a_key)
        nominal = int(pattern_value(config, 'rank_pattern', 'r', name))
        if nominal != a.shape[0]:
            raise ValueError(f'{name}: config rank {nominal} != tensor rank {a.shape[0]}')
        alpha = float(pattern_value(config, 'alpha_pattern', 'lora_alpha', name))
        divisor = math.sqrt(nominal) if config.get('use_rslora') else nominal
        scale = alpha / divisor
        ca, cb = a[:rank].contiguous(), b[:, :rank].contiguous()
        assert ca.shape == (rank, a.shape[1]) and cb.shape == (b.shape[0], rank)
        x = torch.randn(3, a.shape[1], generator=generator, dtype=torch.float64)
        want = scale * ((x @ a.double().T) @ b.double().T)
        got = scale * ((x @ ca.double().T) @ cb.double().T)
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)
        error = float((got - want).abs().max())
        if rank:
            new_alpha = scale * (math.sqrt(rank) if config.get('use_rslora') else rank)
            cfg['target_modules'].append(name)
            # Escape full module names because rank/alpha patterns are regexes.
            cfg['rank_pattern'][re.escape(name)] = rank
            cfg['alpha_pattern'][re.escape(name)] = new_alpha
            output[a_key], output[b_key] = ca, cb
        paired.update((a_key, b_key))
        details.append(dict(module=name, a_key=a_key, b_key=b_key, rank=rank,
                            original_rank=nominal, scale=scale,
                            a_shape=list(ca.shape), b_shape=list(cb.shape),
                            max_abs_error_fp64=error,
                            padded_elements=a.numel()+b.numel(),
                            compact_elements=ca.numel()+cb.numel()))
    if set(tensors) != paired:
        raise ValueError('Unrecognized/non-LoRA tensors; refusing to silently drop additional adapter state')
    manifest = dict(base_only=not output, validation_passed=True, layers=details,
                    original_modules=len(keys), active_modules=sum(d['rank'] > 0 for d in details),
                    zero_modules=sum(d['rank'] == 0 for d in details),
                    original_rank_sum=sum(d['original_rank'] for d in details),
                    compact_rank_sum=sum(ranks),
                    padded_elements=sum(d['padded_elements'] for d in details),
                    compact_elements=sum(d['compact_elements'] for d in details))
    return output, cfg, manifest


def export_compact(source, output, ranks):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Output must be a new directory distinct from the source')
    weights = load_file(source / 'adapter_model.safetensors')
    config = json.loads((source / 'adapter_config.json').read_text())
    compact, cfg, manifest = compact_zero_padded(weights, config, ranks)
    with safe_open(source / 'adapter_model.safetensors', framework='pt') as handle:
        metadata = handle.metadata()
    output.mkdir(parents=True)
    if compact:
        save_file(compact, output / 'adapter_model.safetensors', metadata=metadata)
        (output / 'adapter_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    manifest['source'] = str(source.resolve())
    (output / 'compaction.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allocation', type=Path, required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    document = json.loads(args.allocation.read_text())
    ranks = document['allocation'][args.task]['module_ranks']
    manifest = export_compact(args.input, args.output, ranks)
    print(json.dumps({k:v for k,v in manifest.items() if k != 'layers'}, indent=2))


if __name__ == '__main__':
    main()
