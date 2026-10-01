"""Matched Wan2.1-I2V-14B-480P generations for original vs compressed fleet adapters.

The pipeline is assembled from the Comfy-Org BF16 single files plus the official
Diffusers configs/tokenizer. Adapters are loaded unmerged through diffusers/PEFT
(``load_lora_weights``), one at a time, with identical image, prompt, seed, size,
steps and guidance across variants. For each adapter the original variant is the
reference: compressed variants report PSNR to it and the deviation ratio
||v - orig|| / ||base - orig||, where base is the no-LoRA video (0 = identical to
the original adapter, 1 = as far from it as having no adapter at all).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from safetensors.torch import load_file
from transformers import (CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection,
                          T5TokenizerFast, UMT5Config, UMT5EncoderModel)

NEGATIVE = ('Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, '
            'images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, '
            'incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, '
            'misshapen limbs, fused fingers, still picture, messy background, three legs, many people '
            'in the background, walking backwards')


def load_hf(model_cls, config, path, dtype):
    with torch.device('meta'):
        model = model_cls(config)
    state = {k: v for k, v in load_file(path).items() if v.dtype != torch.uint8}  # drop spiece_model blob
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # T5 ties encoder.embed_tokens to shared; nothing else may be absent.
    missing = [k for k in missing if k != 'encoder.embed_tokens.weight']
    # Older CLIP exports persist the position_ids buffer; current transformers recomputes it.
    unexpected = [k for k in unexpected if not k.endswith('position_ids')]
    if missing or unexpected:
        raise RuntimeError(f'{path}: missing={missing[:5]} unexpected={unexpected[:5]}')
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
    # Non-persistent buffers built under the meta device have no data; rebuild them.
    for owner_name, owner in model.named_modules():
        for name, buffer in list(owner.named_buffers(recurse=False)):
            if not buffer.is_meta:
                continue
            if name != 'position_ids':
                raise RuntimeError(f'{path}: unmaterialised buffer {owner_name}.{name}')
            owner.register_buffer(name, torch.arange(buffer.shape[-1]).expand(buffer.shape), persistent=False)
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftover:
        raise RuntimeError(f'{path}: meta tensors remain: {leftover[:5]}')
    return model.to(dtype).eval()


def build_pipeline(model_dir, flow_shift):
    files, cfg = model_dir / 'split_files', model_dir / 'diffusers_config'
    transformer = WanTransformer3DModel.from_single_file(
        str(files / 'diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors'),
        config=str(cfg), subfolder='transformer', torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_single_file(str(files / 'vae/wan_2.1_vae.safetensors'),
                                            config=str(cfg), subfolder='vae', torch_dtype=torch.float32)
    text_encoder = load_hf(UMT5EncoderModel, UMT5Config.from_pretrained(cfg / 'text_encoder'),
                           files / 'text_encoders/umt5_xxl_fp16.safetensors', torch.bfloat16)
    image_encoder = load_hf(CLIPVisionModelWithProjection, CLIPVisionConfig.from_pretrained(cfg / 'image_encoder'),
                            files / 'clip_vision/clip_vision_h.safetensors', torch.float32)
    scheduler = UniPCMultistepScheduler.from_pretrained(cfg / 'scheduler', flow_shift=flow_shift)
    return WanImageToVideoPipeline(
        tokenizer=T5TokenizerFast.from_pretrained(cfg / 'tokenizer'), text_encoder=text_encoder,
        image_encoder=image_encoder, image_processor=CLIPImageProcessor.from_pretrained(cfg / 'image_processor'),
        transformer=transformer, vae=vae, scheduler=scheduler).to('cuda')


def lora_layers(module):
    return sum(1 for m in module.modules() if hasattr(m, 'lora_A') and len(getattr(m, 'lora_A')) > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True, help='JSON list of {adapter, prompt, variants:{name: path}}')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--height', type=int, default=832)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--frames', type=int, default=49)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance', type=float, default=6.0)
    parser.add_argument('--flow-shift', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Refusing to overwrite a prior generation run')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())

    t0 = time.perf_counter()
    pipe = build_pipeline(args.model_dir, args.flow_shift)
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    print(f'pipeline ready in {time.perf_counter()-t0:.0f}s; '
          f'allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB', flush=True)
    image = load_image(str(args.image))
    settings = dict(height=args.height, width=args.width, num_frames=args.frames, num_inference_steps=args.steps,
                    guidance_scale=args.guidance, flow_shift=args.flow_shift, seed=args.seed,
                    negative_prompt=NEGATIVE, image=str(args.image))

    def generate(prompt, tag):
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        start = time.perf_counter()
        frames = pipe(image=image, prompt=prompt, negative_prompt=NEGATIVE, height=args.height, width=args.width,
                      num_frames=args.frames, num_inference_steps=args.steps, guidance_scale=args.guidance,
                      generator=torch.Generator('cuda').manual_seed(args.seed), output_type='np').frames[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        export_to_video(list(frames), str(args.output_dir / f'{tag}.mp4'), fps=16)
        stats = dict(seconds=seconds, resident_gib=resident/2**30, peak_gib=torch.cuda.max_memory_allocated()/2**30)
        return torch.from_numpy(frames).float(), stats

    def compare(video, reference, base):
        mse = float(((video - reference)**2).mean())
        return dict(psnr_vs_original=float('inf') if mse == 0 else 10*np.log10(1/mse),
                    deviation_ratio=float((video - reference).norm() / (base - reference).norm()))

    results = []
    for item in plan:
        # Base video uses the same prompt (trigger words included) without any adapter.
        base, stats = generate(item['prompt'], f"{item['adapter']}__base")
        results.append(dict(adapter=item['adapter'], variant='base', prompt=item['prompt'], **stats))
        print(f"{item['adapter']} base: {stats}", flush=True)
        reference = None
        for variant, path in item['variants'].items():
            name = f"{item['adapter']}_{variant}".replace('-', '_')
            before = torch.cuda.memory_allocated()
            pipe.load_lora_weights(path, adapter_name=name)
            adapter_bytes = torch.cuda.memory_allocated() - before
            layers = lora_layers(pipe.transformer)
            if layers == 0:
                raise RuntimeError(f'{name}: no LoRA layers were injected')
            video, stats = generate(item['prompt'], f"{item['adapter']}__{variant}")
            pipe.delete_adapters(name)
            record = dict(adapter=item['adapter'], variant=variant, path=path, prompt=item['prompt'],
                          lora_layers=layers, adapter_gpu_bytes=adapter_bytes, **stats)
            if reference is None:
                reference = video
                record['deviation_base_vs_original'] = float((base - video).norm() / video.norm())
            else:
                record.update(compare(video, reference, base))
            results.append(record)
            print(f"{item['adapter']} {variant}: " + json.dumps({k: v for k, v in record.items()
                  if k not in ('path', 'prompt')}), flush=True)
            (args.output_dir / 'results.json').write_text(json.dumps(dict(settings=settings, results=results), indent=1))
    print('done', flush=True)


if __name__ == '__main__':
    main()
