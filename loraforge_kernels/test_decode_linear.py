"""Nonzero-adapter numerical, boundary, dispatch and CUDA-graph regressions."""

import copy
import os

import torch
import torch.nn.functional as F
from torch import nn

from .decode_linear import decode_lora_linear
from .fused_linear import FusedLoRALinear


def main():
    torch.manual_seed(93)
    checks = 0
    for dtype in (torch.float16, torch.bfloat16):
        for m, k, n in ((1, 257, 131), (4, 1536, 512), (8, 769, 259)):
            # Deliberately noncontiguous input and factors, with tail masks.
            x = torch.randn(m, k * 2, device='cuda', dtype=dtype)[:, ::2]
            w = torch.randn(n, k * 2, device='cuda', dtype=dtype)[:, ::2] / k**0.5
            for r in (0, 1, 3, 8, 13, 16, 33, 64):
                a = torch.randn(k, r, device='cuda', dtype=dtype).t() / k**0.5
                b = torch.randn(r, n, device='cuda', dtype=dtype).t() / max(r, 1)**0.5
                for scale, bias in ((0.0, None), (-0.7, torch.randn(n * 2, device='cuda', dtype=dtype)[::2])):
                    ref = F.linear(x.double(), w.double(), None if bias is None else bias.double())
                    ref += scale * F.linear(F.linear(x.double(), a.double()), b.double())
                    for bn in (8, 16, 32):
                        got = decode_lora_linear(x, w, bias, a, b, scale, block_n=bn)
                        err = float(torch.linalg.vector_norm(got.double() - ref) / torch.linalg.vector_norm(ref))
                        assert err < (0.002 if dtype == torch.float16 else 0.012), (m,k,n,r,bn,err)
                        if r == 0 or scale == 0:
                            assert torch.equal(got, F.linear(x, w, bias))
                        checks += 1

    # The module route, swap to rank zero, then swap back.
    base = nn.Linear(256, 512).cuda().bfloat16().eval()
    a = torch.randn(3, 256, device='cuda', dtype=torch.bfloat16) / 16
    b = torch.randn(512, 3, device='cuda', dtype=torch.bfloat16) / 3**0.5
    module = FusedLoRALinear(base, a, b, 0.2)
    x = torch.randn(2, 2, 256, device='cuda', dtype=torch.bfloat16)
    os.environ['LORAFORGE_VARIANT'] = 'decode8'
    want = base(x).float() + 0.2 * F.linear(F.linear(x, a), b).float()
    torch.testing.assert_close(module(x).float(), want, rtol=0.03, atol=0.015)
    module.set_factors(a[:0], b[:, :0])
    assert torch.equal(module(x), base(x))
    module.set_factors(a, b)
    module(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module(x)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(output, module(x))
    os.environ.pop('LORAFORGE_VARIANT')

    # PEFT's default B is zero: randomize it or this test misses wrong updates.
    from peft import LoraConfig, get_peft_model
    from .peft_integration import enable_loraforge_peft, peft_plan_report
    reference = get_peft_model(nn.Sequential(copy.deepcopy(base)), LoraConfig(
        r=3, lora_alpha=3, target_modules=['0'])).eval()
    layer = reference.base_model.model[0]
    layer.lora_A['default'].weight.copy_(a)
    layer.lora_B['default'].weight.copy_(b)
    candidate = copy.deepcopy(reference)
    enable_loraforge_peft(candidate, enable_grouping=False)
    os.environ['LORAFORGE_DECODE_VARIANT'] = 'decode8'
    want, got = reference(x), candidate(x)
    torch.testing.assert_close(got, want, rtol=0.04, atol=0.025)
    assert peft_plan_report(candidate)['decode_choices'] == {'decode8': 1}
    candidate_layer = candidate.base_model.model[0]
    candidate_layer.set_scale('default', 0)
    assert torch.equal(candidate(x), base(x))
    candidate_layer.set_scale('default', 1)
    torch.testing.assert_close(candidate(x), want, rtol=0.04, atol=0.025)
    with candidate.disable_adapter():
        torch.testing.assert_close(candidate(x), base(x))
    os.environ.pop('LORAFORGE_DECODE_VARIANT')
    print(f'PASS: {checks} numeric cases; module swap/zero; CUDA replay; nonzero PEFT adapter')


if __name__ == '__main__':
    with torch.inference_mode():
        main()
