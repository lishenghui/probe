"""End-to-end fleet residency on one GPU: original vs compressed adapter fleets.

For each fleet variant (original, b75, b50, ...), in one process holding the full
Wan2.1-I2V pipeline:
  1. load every adapter of the fleet unmerged (PEFT), timing each disk->GPU load
     and measuring the resident bytes the whole fleet adds;
  2. switch between resident adapters (set_adapters) and time it;
  3. time evicting one adapter and re-loading it from pinned host memory and from disk
     (the cost an adapter cache pays on a miss);
  4. run one full generation with the whole fleet resident and record peak memory.
All adapters are removed before the next variant. Results go to residency.json; the
capacity / cache analysis built on them is in analyze_fleet_residency.py.
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


def gib(x):
    return x / 2**30


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--fleet-dir', type=Path, required=True)
    parser.add_argument('--compressed-dir', type=Path, required=True, help='contains b75/<adapter>/*.safetensors, ...')
    parser.add_argument('--variants', nargs='+', default=['original', 'b75', 'b50', 'b25'])
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--active', default='Assassin')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=49)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--attention-backend', default='native')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a prior benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.fleet_dir / 'manifest.json').read_text())
    prompt = json.loads(args.prompts.read_text())[args.active]

    pipe = build_pipeline(args.model_dir, flow_shift=5.0)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.set_attention_backend(args.attention_backend)
    image = load_image(str(args.image))
    torch.cuda.synchronize()
    pipeline_bytes = torch.cuda.memory_allocated()
    report = dict(gpu=torch.cuda.get_device_name(), total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                  pipeline_bytes=pipeline_bytes, attention_backend=args.attention_backend,
                  frames=args.frames, steps=args.steps, active_adapter=args.active, variants={})

    def generate():
        return pipe(image=image, prompt=prompt, negative_prompt=NEGATIVE, height=832, width=480,
                    num_frames=args.frames, num_inference_steps=args.steps, guidance_scale=6.0,
                    generator=torch.Generator('cuda').manual_seed(42), output_type='latent').frames

    # Peak of one generation with no adapter at all: the fleet-free floor.
    torch.cuda.reset_peak_memory_stats()
    seconds, _ = timed(generate)
    report['no_adapter'] = dict(seconds=seconds, peak_bytes=torch.cuda.max_memory_allocated())
    print(f"pipeline {gib(pipeline_bytes):.2f} GiB; no-adapter generation {seconds:.0f}s "
          f"peak {gib(report['no_adapter']['peak_bytes']):.2f} GiB", flush=True)

    for variant in args.variants:
        paths = {}
        for entry in manifest['entries']:
            if variant == 'original':
                paths[entry['name']] = args.fleet_dir / entry['path']
            else:
                paths[entry['name']] = next((args.compressed_dir / variant / entry['name']).glob('*.safetensors'))
        names = {n: f"{variant}_{n}".replace('-', '_') for n in paths}
        before = torch.cuda.memory_allocated()
        loads, file_bytes, directions = {}, 0, 0
        for adapter, path in paths.items():
            file_bytes += path.stat().st_size
            seconds, state = timed(lambda: load_file(path))
            inject, _ = timed(lambda: pipe.load_lora_weights(dict(state), adapter_name=names[adapter]))
            _, ranks, _ = pin_unit_scaling(pipe.transformer, names[adapter], state)
            directions += ranks
            loads[adapter] = dict(read_s=seconds, inject_s=inject)
        fleet_bytes = torch.cuda.memory_allocated() - before

        # Switching among resident adapters.
        order = list(names.values())
        switch, _ = timed(lambda: [pipe.set_adapters(n) for n in order])
        # Cache miss: evict one adapter, reload it from pinned host memory and from disk.
        victim = args.active
        host = {k: v.pin_memory() for k, v in load_file(paths[victim]).items()}
        pipe.delete_adapters(names[victim])
        from_host, _ = timed(lambda: pipe.load_lora_weights(dict(host), adapter_name=names[victim]))
        pin_unit_scaling(pipe.transformer, names[victim], host)
        pipe.delete_adapters(names[victim])
        from_disk, _ = timed(lambda: pipe.load_lora_weights(load_file(paths[victim]), adapter_name=names[victim]))
        pin_unit_scaling(pipe.transformer, names[victim], load_file(paths[victim]))

        pipe.set_adapters(names[args.active])
        torch.cuda.reset_peak_memory_stats()
        seconds, _ = timed(generate)
        peak = torch.cuda.max_memory_allocated()
        report['variants'][variant] = dict(
            adapters=len(paths), fleet_file_bytes=file_bytes, fleet_directions=directions,
            fleet_resident_bytes=fleet_bytes, per_adapter_resident_bytes=fleet_bytes / len(paths),
            total_load_s=sum(v['read_s'] + v['inject_s'] for v in loads.values()),
            switch_resident_s=switch / len(order), miss_reload_from_pinned_host_s=from_host,
            miss_reload_from_disk_s=from_disk, generation_s=seconds, generation_peak_bytes=peak,
            loads=loads)
        print(f"{variant}: {len(paths)} adapters resident {gib(fleet_bytes):.2f} GiB "
              f"(files {file_bytes/1e9:.2f} GB), load all {report['variants'][variant]['total_load_s']:.0f}s, "
              f"switch {1e3*switch/len(order):.1f} ms, miss host {from_host:.2f}s disk {from_disk:.2f}s, "
              f"generation {seconds:.0f}s peak {gib(peak):.2f} GiB", flush=True)
        args.output.write_text(json.dumps(report, indent=1))
        pipe.delete_adapters(list(names.values()))
        torch.cuda.empty_cache()
    print('done', flush=True)


if __name__ == '__main__':
    main()
