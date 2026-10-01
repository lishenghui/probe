"""Validate the PEFT-free multi-adapter path and measure what it can turn into gains.

Phases (one process, Wan2.1-I2V-14B BF16 pipeline):
  1. correctness: PEFT vs MultiLoRA latents for one adapter (original and a compressed file);
  2. heterogeneous batch: one batch with different adapters per sample vs separate runs;
  3. batch scaling without LoRA: seconds per sample-step for batch 1/2/4 at several
     frame counts -- whether larger batches raise throughput at all;
  4. fleet residency and swap with MultiLoRA: resident bytes, load-all time, and the
     cost of a miss reloaded from pinned host memory and from disk, per fleet variant;
  5. heterogeneous-batch overhead: batch of 4 with no LoRA / one adapter / 4 adapters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from diffusers.utils import load_image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_wan_i2v_fleet_pilot import NEGATIVE, build_pipeline, pin_unit_scaling  # noqa: E402
from multi_lora import MultiLoRA  # noqa: E402


def sync_time(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, out


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--fleet-dir', type=Path, required=True)
    parser.add_argument('--compressed-dir', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=['original', 'b75', 'b50', 'b25'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a prior benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.fleet_dir / 'manifest.json').read_text())
    entries = {e['name']: e for e in manifest['entries']}
    prompts = json.loads(args.prompts.read_text())

    def path(variant, name):
        if variant == 'original':
            return args.fleet_dir / entries[name]['path']
        return next((args.compressed_dir / variant / name).glob('*.safetensors'))

    pipe = build_pipeline(args.model_dir, flow_shift=5.0)
    pipe.set_progress_bar_config(disable=True)
    image = load_image(str(args.image))
    report = dict(gpu=torch.cuda.get_device_name())

    def save():
        args.output.write_text(json.dumps(report, indent=1))

    def run(prompt, frames=17, steps=2, batch=1):
        prompt_list = prompt if isinstance(prompt, list) else [prompt] * batch
        return pipe(image=image, prompt=prompt_list, negative_prompt=[NEGATIVE] * len(prompt_list), height=832,
                    width=480, num_frames=frames, num_inference_steps=steps, guidance_scale=6.0,
                    # One seeded generator per sample: sample i gets the same noise as a batch-1 run.
                    generator=[torch.Generator('cuda').manual_seed(42) for _ in prompt_list],
                    output_type='latent').frames

    # 1. PEFT reference latents, then remove PEFT entirely.
    reference = {}
    for variant in ('original', 'b50'):
        state = load_file(path(variant, 'Assassin'))
        pipe.load_lora_weights(dict(state), adapter_name=f'peft_{variant}')
        pin_unit_scaling(pipe.transformer, f'peft_{variant}', state)
        reference[variant] = run(prompts['Assassin']).cpu()
        pipe.delete_adapters(f'peft_{variant}')
    pipe.unload_lora_weights()
    base_latent = run(prompts['Assassin']).cpu()

    multi = MultiLoRA(pipe.transformer)
    report['correctness'] = {}
    for variant in ('original', 'b50'):
        name = f'{variant}:Assassin'
        multi.add(name, load_file(path(variant, 'Assassin')))
        multi.route(name)
        latent = run(prompts['Assassin']).cpu()
        report['correctness'][variant] = dict(
            rel_diff_multi_vs_peft=rel(latent, reference[variant]),
            rel_diff_peft_vs_base=rel(reference[variant], base_latent))
        multi.remove(name)
    print('correctness', report['correctness'], flush=True)
    save()

    # 2. Heterogeneous batch vs separate single-adapter runs.
    pair = ['Assassin', 'Rotate']
    for n in pair:
        multi.add(f'original:{n}', load_file(path('original', n)))
    singles = []
    for n in pair:
        multi.route(f'original:{n}')
        singles.append(run(prompts[n]).cpu())
    multi.route([f'original:{n}' for n in pair])
    mixed = run([prompts[n] for n in pair]).cpu()
    multi.route(None)
    report['heterogeneous_batch'] = {n: dict(rel_diff_vs_single=rel(mixed[i:i+1], singles[i]),
                                             rel_diff_single_vs_other_single=rel(singles[1-i], singles[i]))
                                     for i, n in enumerate(pair)}
    for n in pair:
        multi.remove(f'original:{n}')
    print('heterogeneous', report['heterogeneous_batch'], flush=True)
    save()

    # 3. Batch scaling without LoRA.
    report['batch_scaling'] = []
    run(prompts['Assassin'], frames=17, steps=1)  # warm-up
    for frames in (17, 33, 49):
        for batch in (1, 2, 4):
            torch.cuda.reset_peak_memory_stats()
            try:
                seconds, _ = sync_time(lambda: run(prompts['Assassin'], frames=frames, steps=3, batch=batch))
                row = dict(frames=frames, batch=batch, seconds=seconds, per_sample_step_s=seconds / (3 * batch),
                           peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            except torch.OutOfMemoryError:
                row = dict(frames=frames, batch=batch, oom=True)
            torch.cuda.empty_cache()
            report['batch_scaling'].append(row)
            print('scaling', row, flush=True)
            save()

    # 4. Fleet residency and swaps with MultiLoRA.
    report['residency'] = {}
    for variant in args.variants:
        before = torch.cuda.memory_allocated()
        read_s = copy_s = 0.0
        for name in entries:
            t, state = sync_time(lambda: load_file(path(variant, name)))
            read_s += t
            t, _ = sync_time(lambda: multi.add(f'{variant}:{name}', state))
            copy_s += t
        resident = torch.cuda.memory_allocated() - before
        host = {k: v.pin_memory() for k, v in load_file(path(variant, 'Assassin')).items()}
        multi.remove(f'{variant}:Assassin')
        from_host, _ = sync_time(lambda: multi.add(f'{variant}:Assassin', host))
        multi.remove(f'{variant}:Assassin')
        from_disk, _ = sync_time(lambda: multi.add(f'{variant}:Assassin', load_file(path(variant, 'Assassin'))))
        switch, _ = sync_time(lambda: [multi.route(f'{variant}:{n}') for n in entries])
        report['residency'][variant] = dict(
            adapters=len(entries), resident_bytes=resident, adapter_bytes=sum(
                multi.adapters[f'{variant}:{n}']['bytes'] for n in entries),
            read_all_s=read_s, copy_all_s=copy_s, miss_from_pinned_host_s=from_host,
            miss_from_disk_s=from_disk, route_switch_s=switch / len(entries))
        print('residency', variant, report['residency'][variant], flush=True)
        save()
        if variant != args.variants[-1]:
            for name in entries:
                multi.remove(f'{variant}:{name}')
        torch.cuda.empty_cache()

    # 5. Heterogeneous-batch overhead with the last fleet variant resident.
    last = args.variants[-1]
    four = list(entries)[:4]
    report['hetero_overhead'] = {}
    for label, route in (('no_lora', None), ('same_adapter', f'{last}:{four[0]}'),
                         ('four_adapters', [f'{last}:{n}' for n in four])):
        multi.route(route)
        seconds, _ = sync_time(lambda: run(prompts['Assassin'], frames=49, steps=3, batch=4))
        report['hetero_overhead'][label] = dict(seconds=seconds, per_sample_step_s=seconds / 12)
        print('hetero overhead', label, report['hetero_overhead'][label], flush=True)
    multi.route(None)
    save()
    print('done', flush=True)


if __name__ == '__main__':
    main()
