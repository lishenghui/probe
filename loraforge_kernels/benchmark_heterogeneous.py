"""Validate real FRA-0 compaction, then measure prefill/decode against both ceilings.

Workload and variants are fixed before measurement. No timings are produced
until live module shapes/scales and padded-vs-compact outputs have passed.
"""

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from peft.tuners.lora.layer import Linear as LoraLinear
from transformers import AutoModelForCausalLM

from .compact_adapter import module_name
from .peft_integration import enable_loraforge_peft, peft_plan_report
from .prepare_fra0_adapters import sha256
from .static_decode import StaticDecodeRunner


def audit(model, manifest, compact):
    records = []
    for detail in manifest['layers']:
        layer = model.get_submodule(detail['module'])
        rank = detail['rank'] if compact else detail['original_rank']
        if rank == 0:
            assert isinstance(layer, torch.nn.Linear) and not isinstance(layer, LoraLinear)
            records.append(dict(module=detail['module'], rank=0, plain_base=True, elements=0))
            continue
        assert isinstance(layer, LoraLinear), detail['module']
        a, b = layer.lora_A['default'].weight, layer.lora_B['default'].weight
        assert a.shape == (rank, layer.in_features), (detail['module'], a.shape, rank)
        assert b.shape == (layer.out_features, rank), (detail['module'], b.shape, rank)
        assert a.dtype == b.dtype == torch.bfloat16
        assert abs(layer.scaling['default'] - detail['scale']) < 1e-12
        records.append(dict(module=detail['module'], rank=rank, a_shape=list(a.shape),
                            b_shape=list(b.shape), scale=layer.scaling['default'],
                            elements=a.numel()+b.numel()))
    total = sum(r['elements'] for r in records)
    assert total == manifest['compact_elements' if compact else 'padded_elements']
    assert sum(isinstance(m, LoraLinear) for m in model.modules()) == sum(r['rank'] > 0 for r in records)
    return dict(passed=True, adapter_elements=total, adapter_bf16_bytes=2*total,
                active_modules=sum(r['rank'] > 0 for r in records), layers=records)


def input_ids(model, batch, prefix):
    generator = torch.Generator(device='cuda').manual_seed(8721 + batch + prefix)
    return torch.randint(3, model.config.vocab_size, (batch, prefix), generator=generator, device='cuda')


def validation_logits(model, ids, steps):
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
    logits = [output.logits[:, -1].float().cpu()]
    cache = output.past_key_values
    token = torch.full((ids.shape[0], 1), 42, device='cuda', dtype=torch.long)
    for _ in range(steps):
        output = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        logits.append(output.logits[:, -1].float().cpu())
    return torch.stack(logits)


def compare(got, reference):
    error = got - reference
    relative_l2 = float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference).clamp_min(1e-8))
    max_normalized = float(error.abs().max() / reference.abs().max().clamp_min(1e-8))
    # Declared before measurement: bf16 full-model tolerance, in addition to
    # the much stricter per-layer fp64 algebraic equivalence in compaction.json.
    assert relative_l2 <= 0.02 and max_normalized <= 0.04, (relative_l2, max_normalized)
    return dict(relative_l2=relative_l2, max_normalized_error=max_normalized, passed=True)


def attach(base, path, kernel=False):
    model = PeftModel.from_pretrained(base, str(path), autocast_adapter_dtype=False).eval()
    if kernel:
        enable_loraforge_peft(model, enable_grouping=False, cast_adapter_dtype=False)
    return model


def detach(model):
    base = model.unload()
    assert not any(isinstance(m, LoraLinear) for m in base.modules())
    return base


def measured_request(runner, ids, graph):
    result = torch.empty((ids.shape[0], ids.shape[1]+runner.new_tokens), device='cuda', dtype=torch.long)
    torch.cuda.synchronize()
    start = time.perf_counter()
    runner._prefill(ids)
    result[:, :runner.prefix].copy_(ids)
    result[:, runner.prefix:runner.prefix+1].copy_(runner.token)
    torch.cuda.synchronize()
    prefill_done = time.perf_counter()
    for step in range(1, runner.new_tokens):
        runner.graph.replay() if graph else runner._step()
        result[:, runner.prefix+step:runner.prefix+step+1].copy_(runner.token)
    torch.cuda.synchronize()
    done = time.perf_counter()
    return dict(prefill_ms=1000*(prefill_done-start), decode_ms=1000*(done-prefill_done),
                total_ms=1000*(done-start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=['arc_challenge','story_cloze','sst2'])
    parser.add_argument('--batches', nargs='+', type=int, default=[1,4])
    parser.add_argument('--prefixes', nargs='+', type=int, default=[128,512])
    parser.add_argument('--new-tokens', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    import os
    # Graph-based candidate selection is fixed across both measurements. The
    # eager replay of that same plan isolates scheduling from kernel choice.
    os.environ['LORAFORGE_TUNE_MODE'] = 'graph'
    os.environ['LORAFORGE_ENABLE_TILED'] = '0'
    os.environ['LORAFORGE_ENABLE_DECODE_LINEAR'] = '1'
    gpu = torch.cuda.get_device_name()
    assert 'GH200' in gpu, gpu
    args.output.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               device_map='cuda', local_files_only=True).eval()
    validations = {}
    shapes = [(batch, prefix) for batch in args.batches for prefix in args.prefixes]
    manifests = {task: json.loads((args.prepared/task/'compact/compaction.json').read_text()) for task in args.tasks}
    variants = ('uncompressed', 'padded', 'compact', 'compact_kernel')

    def path_for(task, variant):
        return args.sources/task if variant == 'uncompressed' else args.prepared/task/('padded' if variant == 'padded' else 'compact')

    # Phase 1: validate everything before starting the timing phase.
    for task in args.tasks:
        manifest = manifests[task]
        model = attach(base, path_for(task, 'padded'))
        padded_audit = audit(base, manifest, False)
        references = {f'{b}x{p}': validation_logits(model, input_ids(base,b,p), 3) for b,p in shapes}
        base = detach(model)
        del model
        task_validation = dict(padded=padded_audit, variants={})
        for variant in ('compact', 'compact_kernel'):
            model = attach(base, path_for(task, variant), variant == 'compact_kernel')
            shape_audit = audit(base, manifest, True)
            output_checks = {}
            for batch, prefix in shapes:
                key = f'{batch}x{prefix}'
                ids = input_ids(base,batch,prefix)
                got = validation_logits(model, ids, 3)
                output_checks[key] = compare(got, references[key])
                runner = StaticDecodeRunner(model, ids, args.new_tokens)
                graph_error = runner.validate(ids, tolerance=0.01)
                eager, captured = runner(ids, replay=False), runner(ids)
                assert torch.equal(eager, captured), (task, variant, key, 'graph trajectory mismatch')
                output_checks[key].update(graph_logit_relative_l2=graph_error, graph_tokens_equal=True)
                del runner
            task_validation['variants'][variant] = dict(shape_audit=shape_audit, output_checks=output_checks)
            base = detach(model)
            del model
            gc.collect()
        task_validation['passed'] = True
        validations[task] = task_validation
        (args.output/'validation.json').write_text(json.dumps(validations, indent=2)+'\n')
        print(f'VALIDATED {task}: physical elements {manifest["padded_elements"]} -> {manifest["compact_elements"]}; zero modules {manifest["zero_modules"]}', flush=True)
    assert all(v['passed'] for v in validations.values())

    # Phase 2: same workload and timing protocol for pure base and all adapters.
    rows = []
    metadata = dict(gpu=gpu, dtype='bfloat16', torch=torch.__version__,
                    input_tokens=args.prefixes, batches=args.batches, output_tokens=args.new_tokens,
                    repeats=args.repeats, lora_grouping=False, validation_completed_before_timing=True,
                    validation_sha256=sha256(args.output/'validation.json'),
                    workload='Fixed random input IDs, greedy fixed-length output; real FRA-0 adapters',
                    timing='Wall clock synchronized at prefill/decode boundary; prefill includes StaticCache reset; decode generates remaining output_tokens-1 tokens')

    def measure(model, task, variant):
        if task != 'base':
            audit(base, manifests[task], variant.startswith('compact'))
        for batch,prefix in shapes:
            ids = input_ids(base,batch,prefix)
            runner = StaticDecodeRunner(model, ids, args.new_tokens)
            for graph in (False,True):
                measured_request(runner,ids,graph)
            samples = {False: [], True: []}
            rng = random.Random(415)
            for _ in range(args.repeats):
                order = [False,True]
                rng.shuffle(order)
                for graph in order:
                    samples[graph].append(measured_request(runner,ids,graph))
            for graph in (False,True):
                summary = {k:statistics.median(x[k] for x in samples[graph]) for k in ('prefill_ms','decode_ms','total_ms')}
                row = dict(task=task, variant=variant, batch=batch, prefix=prefix,
                           mode='graph' if graph else 'eager', **summary,
                           total_tokens_s=batch*args.new_tokens*1000/summary['total_ms'],
                           decode_tokens_s=batch*(args.new_tokens-1)*1000/summary['decode_ms'],
                           samples=samples[graph])
                if task != 'base':
                    row.update(active_lora_modules=manifests[task]['active_modules'] if variant.startswith('compact') else manifests[task]['original_modules'],
                               adapter_elements=manifests[task]['compact_elements'] if variant.startswith('compact') else manifests[task]['padded_elements'])
                if variant == 'compact_kernel':
                    row['plan'] = peft_plan_report(model)
                rows.append(row)
                (args.output/'timings.json').write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('samples','plan')}), flush=True)
            del runner
            gc.collect()

    measure(base, 'base', 'base_start')
    for task in args.tasks:
        order = list(variants)
        random.Random(781 + args.tasks.index(task)).shuffle(order)
        for variant in order:
            model = attach(base, path_for(task,variant), variant == 'compact_kernel')
            measure(model,task,variant)
            base = detach(model)
            del model
            gc.collect()
    measure(base, 'base', 'base_end')


if __name__ == '__main__':
    with torch.inference_mode():
        main()
