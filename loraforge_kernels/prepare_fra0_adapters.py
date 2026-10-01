"""Reproduce FRA-0 padded references read-only, then export compact adapters."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .compact_adapter import export_compact


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=['arc_challenge','story_cloze','sst2'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Import the finished paper's reference function without writing pycache or
    # modifying any file in the submodule. Keep its exact numerical algorithm.
    sys.dont_write_bytecode = True
    reference = Path(__file__).resolve().parents[1] / 'ruller-paper/experiments/rq3/compress_adapter.py'
    spec = importlib.util.spec_from_file_location('fra0_reference_readonly', reference)
    compressor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compressor)
    allocation = json.loads(args.allocation.read_text())
    records = []
    for task in args.tasks:
        source = args.sources / task
        target = args.output / task
        if target.exists():
            raise ValueError(f'Refusing to overwrite {target}')
        ranks = allocation['allocation'][task + '_10templates']['module_ranks']
        weights = load_file(source / 'adapter_model.safetensors')
        keys = sorted(k for k in weights if k.endswith('.lora_A.weight'))
        assert len(keys) == len(ranks)
        padded = dict(weights)
        for a_key, rank in zip(keys, ranks):
            b_key = a_key.replace('.lora_A.weight', '.lora_B.weight')
            a, b, retained, _ = compressor.truncated_factors(weights[a_key], weights[b_key], 0.0, fixed_rank=rank)
            assert retained == rank
            padded[a_key], padded[b_key] = a, b
        padded_dir = target / 'padded'
        padded_dir.mkdir(parents=True)
        save_file(padded, padded_dir / 'adapter_model.safetensors')
        (padded_dir / 'adapter_config.json').write_bytes((source / 'adapter_config.json').read_bytes())
        manifest = export_compact(padded_dir, target / 'compact', ranks)
        record = dict(task=task, source=str(source.resolve()),
                      source_sha256=sha256(source / 'adapter_model.safetensors'),
                      padded_sha256=sha256(padded_dir / 'adapter_model.safetensors'),
                      compact_sha256=sha256(target / 'compact/adapter_model.safetensors'),
                      reference_code_sha256=sha256(reference),
                      allocation=str(args.allocation.resolve()), allocation_sha256=sha256(args.allocation),
                      **{k:v for k,v in manifest.items() if k not in ('source','layers')})
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output / 'preparation.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
