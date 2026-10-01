"""Paired full-Linear rank sweep, with eager and CUDA-graph measurements."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .decode_linear import decode_lora_linear
from .fused_lora import fused_lora, tiled_lora


def paired_times(functions, repeats, graph, inner=20):
    runners = {}
    for name, fn in functions.items():
        for _ in range(5):
            fn()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                for _ in range(inner):
                    fn()
            runners[name] = capture.replay
        else:
            runners[name] = fn
    torch.cuda.synchronize()
    samples = {name: [] for name in runners}
    rng = random.Random(17)
    names = list(runners)
    for _ in range(repeats):
        rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize()
            start = time.perf_counter()
            if graph:
                runners[name]()
            else:
                for _ in range(inner):
                    runners[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1e6 / inner)
    return {name: statistics.median(values) for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shapes', nargs='+', default=['1536,1536', '2560,2560', '2560,9728', '9728,2560'])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 4, 8])
    parser.add_argument('--ranks', nargs='+', type=int, default=[0, 1, 4, 8, 16, 32])
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for shape in args.shapes:
        k, n = map(int, shape.split(','))
        for m in args.batches:
            x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
            w = torch.randn(n, k, device='cuda', dtype=x.dtype) / k**0.5
            bias = torch.randn(n, device='cuda', dtype=x.dtype) * 0.1
            for r in args.ranks:
                a = torch.randn(r, k, device='cuda', dtype=x.dtype) / k**0.5
                b = torch.randn(n, r, device='cuda', dtype=x.dtype) / max(r, 1)**0.5
                scale = 0.3
                funcs = {'base': lambda: F.linear(x, w, bias)}
                funcs['direct'] = lambda: F.linear(x, w, bias).add_(F.linear(F.linear(x, a), b), alpha=scale)
                if r:
                    funcs['row'] = lambda: fused_lora(x, a, b, F.linear(x, w, bias), scale)
                    funcs['tiled'] = lambda: tiled_lora(x, a, b, F.linear(x, w, bias), scale)
                for bn in (8, 16, 32):
                    funcs[f'decode{bn}'] = lambda bn=bn: decode_lora_linear(x, w, bias, a, b, scale, block_n=bn)
                reference = F.linear(x.double(), w.double(), bias.double()) + scale * F.linear(F.linear(x.double(), a.double()), b.double())
                errors = {}
                for name, fn in funcs.items():
                    if name == 'base':
                        continue
                    error = float(torch.linalg.vector_norm(fn().double() - reference) / torch.linalg.vector_norm(reference))
                    if error > 0.015:
                        raise AssertionError((shape, m, r, name, error))
                    errors[name] = error
                eager = paired_times(funcs, args.repeats, False)
                graph = paired_times(funcs, args.repeats, True)
                row = dict(m=m, k=k, n=n, rank=r, eager_us=eager, graph_us=graph, relative_l2=errors)
                rows.append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    with torch.inference_mode():
        main()
