"""Audit historical fleet allocations and optionally measure factor storage on GPU.

No compression, model download, model inference, or quality evaluation is done.
The GPU probe reconstructs nominal-shape zero padding from already compact
factors, then compares tensor bytes and allocator bytes at identical BF16 dtype.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path


def read(path, receipts):
    raw = path.read_bytes()
    receipts[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def fleet_audit(root, receipts):
    rows = []
    configs = {
        'land': ('LoRA Land', 12, 8, 'land12'),
        'cts': ('Lots-of-LoRAs', 25, 16, 'cts25'),
        'lorare': ('LoRARetriever', 41, 8, 'lorare'),
    }
    for pool, (label, count, nominal_rank, prefix) in configs.items():
        for path in sorted((root / 'fra_clean_alloc').glob(f'{pool}_fra_b*.json')):
            doc = read(path, receipts)
            assert len(doc['allocation']) == count
            params = full_params = active = modules = rank_sum = 0
            supports = []
            rank_vectors = []
            reference_keys = None
            for task, allocation in sorted(doc['allocation'].items()):
                curve = read(root / f'functional_dp0_output_{prefix}_{task}.json', receipts)
                keys = curve['module_keys']
                ranks = allocation['module_ranks']
                assert len(keys) == len(ranks)
                if reference_keys is None:
                    reference_keys = keys
                assert keys == reference_keys, 'Mixed layouts need key-aligned support analysis'
                assert sum(ranks) == allocation['k']
                supports.append([r > 0 for r in ranks])
                rank_vectors.append(ranks)
                for key, rank in zip(keys, ranks):
                    assert 0 <= rank <= nominal_rank
                    # Architecture dimensions from the historical paper's fleet table.
                    if pool == 'lorare' or '.q_proj.' in key:
                        cost = 8192
                    elif '.k_proj.' in key or '.v_proj.' in key:
                        cost = 5120
                    else:
                        raise ValueError(f'Unsupported target: {key}')
                    params += rank * cost
                    full_params += nominal_rank * cost
                    active += rank > 0
                    modules += 1
                    rank_sum += rank
            assert rank_sum == doc['spent']
            zero_counts = [sum(not s[j] for s in supports) for j in range(len(reference_keys))]
            # Exact expectation for uniformly sampled DISTINCT adapters, not traffic evidence.
            expected_skip = {}
            for distinct in sorted({1, 4, 8, count}):
                expected_skip[str(distinct)] = sum(
                    math.comb(z, distinct) / math.comb(count, distinct)
                    if z >= distinct else 0 for z in zero_counts
                ) / len(reference_keys)
            rows.append(dict(
                fleet=label, allocation=str(path), n=count, budget=doc['budget'],
                spent=rank_sum, nominal_rank=nominal_rank,
                full_parameters=full_params, compact_parameters=params,
                full_bf16_mib=full_params * 2 / 2**20,
                compact_bf16_mib=params * 2 / 2**20,
                saved_bf16_mib=(full_params-params) * 2 / 2**20,
                parameter_reduction=1-params/full_params,
                # Illustrative floor: 7B parameters, BF16, whole fleet resident.
                # KV, activations, allocator reserves and workspaces are excluded.
                saved_fraction_of_7b_base_plus_fleet=(full_params-params)/(7e9+full_params),
                active_branches=active, original_branches=modules,
                removed_branch_fraction=1-active/modules,
                maximum_retained_rank=max(max(ranks) for ranks in rank_vectors),
                expected_skipped_module_fraction_uniform_distinct=expected_skip,
            ))
    assert len(rows) == 9
    return rows


def historical_timing(source, receipts):
    directory = source / 'artifacts/kernel_bench/heterogeneous-fra0/measured-b3191'
    summary = read(directory / 'summary.json', receipts)
    validation = read(directory / 'validation.json', receipts)
    timings = read(directory / 'timings.json', receipts)
    rows = []
    for row in summary:
        if row['mode'] != 'graph' or row['prefix'] != 128:
            continue
        item = dict(row)
        if row['variant'] in ('padded', 'compact'):
            task_check = validation[row['task']]
            check = (task_check['padded'] if row['variant'] == 'padded'
                     else task_check['variants']['compact']['shape_audit'])
            item['validated_bf16_bytes'] = check['adapter_bf16_bytes']
            item['validated_active_modules'] = check['active_modules']
        rows.append(item)
    return dict(metadata=timings['metadata'], rows=rows)


def factor_memory(source, receipts):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    torch.empty(1, device='cuda')
    results = []
    directory = source / 'artifacts/kernel_bench/heterogeneous-fra0/rebuilt-20260908'

    def measure(cpu_factors):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        reserve_before = torch.cuda.memory_reserved()
        gpu_factors = {k: v.to(device='cuda', dtype=torch.bfloat16) for k, v in cpu_factors.items()}
        torch.cuda.synchronize()
        result = dict(tensor_bytes=sum(v.numel()*v.element_size() for v in gpu_factors.values()),
                      allocated_delta=torch.cuda.memory_allocated()-before,
                      reserved_delta=torch.cuda.memory_reserved()-reserve_before)
        del gpu_factors
        gc.collect()
        torch.cuda.empty_cache()
        return result

    for task in ('arc_challenge', 'story_cloze', 'sst2'):
        manifest = read(directory / task / 'compaction.json', receipts)
        weights_path = directory / task / 'adapter_model.safetensors'
        receipts[str(weights_path)] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        compact = load_file(weights_path)
        padded = {}
        for layer in manifest['layers']:
            rank, nominal = layer['rank'], layer['original_rank']
            a = torch.zeros(nominal, layer['a_shape'][1], dtype=torch.bfloat16)
            b = torch.zeros(layer['b_shape'][0], nominal, dtype=torch.bfloat16)
            if rank:
                a[:rank].copy_(compact[layer['a_key']])
                b[:, :rank].copy_(compact[layer['b_key']])
                assert torch.equal(a[:rank], compact[layer['a_key']].to(torch.bfloat16))
                assert torch.equal(b[:, :rank], compact[layer['b_key']].to(torch.bfloat16))
            assert not torch.count_nonzero(a[rank:])
            assert not torch.count_nonzero(b[:, rank:])
            padded[layer['a_key']], padded[layer['b_key']] = a, b
        compact_result, padded_result = measure(compact), measure(padded)
        assert compact_result['tensor_bytes'] == manifest['compact_elements'] * 2
        assert padded_result['tensor_bytes'] == manifest['padded_elements'] * 2
        results.append(dict(task=task, dtype='bfloat16', compact=compact_result,
                            reconstructed_padded=padded_result,
                            file_bytes=weights_path.stat().st_size,
                            source_dtypes=sorted({str(v.dtype) for v in compact.values()}),
                            active_modules=manifest['active_modules'],
                            original_modules=manifest['original_modules']))
        del compact, padded, a, b
    return dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=results,
                scope='Factor allocations only. No base, KV cache, PEFT loader or inference. '
                      'Padding reconstructed from preserved compact factors; no new compression.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-memory', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing audit')
    receipts = {}
    receipts[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(job_id=os.environ.get('SLURM_JOB_ID'),
                  scope='Historical evidence audit; optional fresh factor-only GPU memory measurement',
                  fleet=fleet_audit(args.results, receipts),
                  historical_timing=historical_timing(args.source, receipts))
    if args.gpu_memory:
        report['gpu_memory'] = factor_memory(args.source, receipts)
    report['source_sha256'] = receipts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for row in report['fleet']:
        print(f"{row['fleet']:15s} B={row['budget']:5d}: "
              f"{row['full_bf16_mib']:.2f}->{row['compact_bf16_mib']:.2f} MiB; "
              f"branches removed={row['removed_branch_fraction']:.1%}; "
              f"7B+fleet reduction={row['saved_fraction_of_7b_base_plus_fleet']:.2%}")
    if 'gpu_memory' in report:
        for row in report['gpu_memory']['rows']:
            print(row)
    print(f'Audit saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
