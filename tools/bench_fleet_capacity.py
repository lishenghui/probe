"""What a smaller resident fleet buys under a fixed GPU memory cap (measured by OOM).

For each fleet variant the whole fleet (49 adapters) is resident and unmerged, the
process is capped with torch.cuda.set_per_process_memory_fraction to emulate a
smaller GPU, and we search for the largest workload that completes without OOM:
  * max frames (batch 1), and
  * max batch size (videos per call) at a fixed frame count.
Each probe is a real end-to-end call including VAE decode; peak memory does not
depend on the step count, so probes use few steps. Optional --fp8-storage keeps the
DiT weights in float8 (compute in bf16), the usual low-memory deployment.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--fleet-dir', type=Path, required=True)
    parser.add_argument('--compressed-dir', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=['none', 'original', 'b75', 'b50', 'b25'])
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--active', default='Assassin')
    parser.add_argument('--cap-gib', type=float, required=True)
    parser.add_argument('--fp8-storage', action='store_true')
    parser.add_argument('--frames', type=int, nargs='+', default=[49, 65, 81, 97, 113, 129, 145, 161, 177, 193])
    parser.add_argument('--batch-frames', type=int, default=49)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a prior benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.fleet_dir / 'manifest.json').read_text())
    prompt = json.loads(args.prompts.read_text())[args.active]

    pipe = build_pipeline(args.model_dir, flow_shift=5.0)
    if args.fp8_storage:
        pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    image = load_image(str(args.image))
    total = torch.cuda.get_device_properties(0).total_memory
    if args.cap_gib * 2**30 > total:
        raise ValueError('cap exceeds device memory')
    torch.cuda.set_per_process_memory_fraction(args.cap_gib * 2**30 / total)
    torch.cuda.empty_cache()
    report = dict(gpu=torch.cuda.get_device_name(), cap_gib=args.cap_gib, fp8_storage=args.fp8_storage,
                  steps=args.steps, pipeline_bytes=torch.cuda.memory_allocated(), variants={})
    print(f"cap {args.cap_gib} GiB, fp8_storage={args.fp8_storage}, pipeline "
          f"{torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    def runs(frames, batch):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            pipe(image=image, prompt=prompt, negative_prompt=NEGATIVE, height=832, width=480,
                 num_frames=frames, num_inference_steps=args.steps, guidance_scale=6.0,
                 num_videos_per_prompt=batch, generator=torch.Generator('cuda').manual_seed(42), output_type='np')
            torch.cuda.synchronize()
            return dict(ok=True, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                        seconds=time.perf_counter() - start)
        except torch.OutOfMemoryError:
            return dict(ok=False)
        finally:
            torch.cuda.empty_cache()

    for variant in args.variants:
        names = []
        before = torch.cuda.memory_allocated()
        try:
            for entry in ([] if variant == 'none' else manifest['entries']):
                path = (args.fleet_dir / entry['path'] if variant == 'original'
                        else next((args.compressed_dir / variant / entry['name']).glob('*.safetensors')))
                name = f"{variant}_{entry['name']}".replace('-', '_')
                state = load_file(path)
                pipe.load_lora_weights(dict(state), adapter_name=name)
                pin_unit_scaling(pipe.transformer, name, state)
                names.append(name)
            fleet_ok = True
        except torch.OutOfMemoryError:
            fleet_ok = False
        resident = torch.cuda.memory_allocated() - before
        result = dict(adapters_requested=0 if variant == 'none' else len(manifest['entries']),
                      adapters_loaded=len(names), fleet_fits=fleet_ok, fleet_resident_gib=resident / 2**30,
                      frames={}, batch={})
        if fleet_ok:
            if names:
                pipe.set_adapters(f"{variant}_{args.active}".replace('-', '_'))
            for frames in args.frames:  # ascending; stop at the first OOM
                result['frames'][frames] = runs(frames, 1)
                print(f"{variant}: frames {frames} -> {result['frames'][frames]}", flush=True)
                if not result['frames'][frames]['ok']:
                    break
            for batch in args.batches:
                result['batch'][batch] = runs(args.batch_frames, batch)
                print(f"{variant}: batch {batch}x{args.batch_frames}f -> {result['batch'][batch]}", flush=True)
                if not result['batch'][batch]['ok']:
                    break
        ok_frames = [f for f, r in result['frames'].items() if r['ok']]
        ok_batch = [b for b, r in result['batch'].items() if r['ok']]
        result['max_frames'] = max(ok_frames, default=0)
        result['max_batch'] = max(ok_batch, default=0)
        report['variants'][variant] = result
        print(f"== {variant}: fleet {'fits' if fleet_ok else 'DOES NOT FIT'} ({len(names)} loaded, "
              f"{resident/2**30:.2f} GiB), max frames {result['max_frames']}, "
              f"max batch@{args.batch_frames}f {result['max_batch']}", flush=True)
        args.output.write_text(json.dumps(report, indent=1))
        if names:
            pipe.delete_adapters(names)
        torch.cuda.empty_cache()
    print('done', flush=True)


if __name__ == '__main__':
    main()
