"""Full local causal-LM generation with synthetic, nonzero LoRAs.

This isolates inference overhead across ranks; it is not a compression-quality
experiment. Uses the same loaded base weights, fixed token counts, and dtype.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from . import peft_integration as bridge
from .fused_linear import _time_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--new-tokens', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--enable-tiled', action='store_true', help='Include the existing tiled sidecar in optimized selection')
    args = parser.parse_args()
    os.environ['LORAFORGE_TUNE_MODE'] = 'graph' if args.cuda_graph else 'eager'
    torch.manual_seed(7)
    raw = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                             device_map='cuda', local_files_only=True).eval()
    rows = []
    graph_timer = bridge._cuda_median_us
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def measure(model, label, rank):
        for batch in args.batches:
            inputs = torch.full((batch, 64), 42, device='cuda', dtype=torch.long)
            inputs[:, 0] = model.config.bos_token_id or 1
            mask = torch.ones_like(inputs)
            runner = None
            validation = None
            if args.cuda_graph:
                from .static_decode import StaticDecodeRunner
                runner = StaticDecodeRunner(model, inputs, args.new_tokens)
                eager = runner(inputs, replay=False)
                captured = runner(inputs)
                validation = dict(greedy_tokens_equal=torch.equal(eager, captured),
                                  logit_relative_l2=runner.validate(inputs))

            def run():
                if runner is not None:
                    return runner(inputs)
                return model.generate(input_ids=inputs, attention_mask=mask,
                                      do_sample=False, min_new_tokens=args.new_tokens,
                                      max_new_tokens=args.new_tokens, pad_token_id=0)
            run()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                output = run()
                torch.cuda.synchronize()
                assert output.shape == (batch, 64 + args.new_tokens)
                samples.append(time.perf_counter() - start)
            median = statistics.median(samples)
            row = dict(variant=label, rank=rank, batch=batch, seconds=median,
                       tokens_per_second=batch*args.new_tokens/median, samples=samples,
                       graph_validation=validation)
            if label in ('previous', 'optimized'):
                row['plan'] = bridge.peft_plan_report(model)
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k != 'plan'}), flush=True)
            args.output.write_text(json.dumps(dict(model=args.model, gpu=torch.cuda.get_device_name(),
                torch=torch.__version__, workload='synthetic nonzero LoRA; 64 input tokens; fixed-length generation',
                new_tokens=args.new_tokens, grouping=False, cuda_graph=args.cuda_graph,
                enable_tiled=args.enable_tiled, rows=rows), indent=2)+'\n')

    measure(raw, 'base_start', 0)
    model = get_peft_model(raw, LoraConfig(r=max(args.ranks), lora_alpha=max(args.ranks),
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])).eval()
    layers = [m for m in model.modules() if isinstance(m, bridge.LoraLinear)]
    factors = []
    for layer in layers:
        a = layer.lora_A['default'].weight
        b = layer.lora_B['default'].weight
        factors.append((torch.randn_like(a, dtype=torch.bfloat16) / a.shape[1]**0.5,
                        torch.randn_like(b, dtype=torch.bfloat16) / b.shape[1]**0.5))
    for rank in args.ranks:
        for layer, (a, b) in zip(layers, factors):
            layer.lora_A['default'].weight = torch.nn.Parameter(a[:rank].contiguous(), requires_grad=False)
            layer.lora_B['default'].weight = torch.nn.Parameter(b[:, :rank].contiguous(), requires_grad=False)
            layer.scaling['default'] = 0.1
            layer.r['default'] = rank
        for variant in ('peft', 'previous', 'optimized'):
            os.environ['LORAFORGE_ENABLE_TILED'] = '1' if variant == 'optimized' and args.enable_tiled else '0'
            for layer in layers:
                if hasattr(layer, '_loraforge_original_forward'):
                    layer.forward = layer._loraforge_original_forward
                    del layer._loraforge_original_forward
            bridge._DECODE_CHOICE_CACHE.clear()
            if variant != 'peft':
                os.environ['LORAFORGE_ENABLE_DECODE_LINEAR'] = '1' if variant == 'optimized' else '0'
                bridge._cuda_median_us = graph_timer if variant == 'optimized' else lambda fn: _time_ms(fn)*1000
                bridge.enable_loraforge_peft(model, enable_grouping=False)
            measure(model, variant, rank)
    for layer in layers:
        layer.forward = layer._loraforge_original_forward
    raw = model.unload()
    measure(raw, 'base_end', 0)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
