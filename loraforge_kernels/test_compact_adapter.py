"""Compaction must change physical storage while preserving real PEFT outputs."""

import copy
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from safetensors.torch import save_file
from torch import nn

from .compact_adapter import compact_zero_padded, export_compact


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(7, 5) for _ in range(3)])

    def forward(self, x):
        return sum(layer(x) for layer in self.blocks)


def main():
    import json
    torch.manual_seed(83)
    for rslora in (False, True):
        config = dict(peft_type='LORA', r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=['0','1','2'], bias='none', use_rslora=rslora,
                      alpha_pattern={r'blocks\.1': 5.5})
        ranks = [0,1,3]
        tensors = {}
        for i,rank in enumerate(ranks):
            a, b = torch.zeros(8,7), torch.zeros(5,8)
            a[:rank].normal_()
            b[:, :rank].normal_()
            tensors[f'base_model.model.blocks.{i}.lora_A.weight'] = a
            tensors[f'base_model.model.blocks.{i}.lora_B.weight'] = b
        compact, cfg, manifest = compact_zero_padded(tensors,config,ranks)
        assert manifest['compact_elements'] == 4*(7+5)
        assert manifest['padded_elements'] == 3*8*(7+5)
        assert manifest['zero_modules'] == 1 and manifest['active_modules'] == 2
        assert len(compact) == 4
        base = Toy().double().eval()
        reference = get_peft_model(copy.deepcopy(base), LoraConfig(**config)).eval()
        set_peft_model_state_dict(reference,tensors)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            save_file(tensors,source/'adapter_model.safetensors')
            (source/'adapter_config.json').write_text(json.dumps(config))
            export_compact(source,root/'compact',ranks)
            candidate = PeftModel.from_pretrained(copy.deepcopy(base),root/'compact',
                                                   autocast_adapter_dtype=False, torch_device='cpu').eval()
            for i,rank in enumerate(ranks):
                layer = candidate.base_model.model.blocks[i]
                if rank:
                    assert layer.lora_A['default'].weight.shape == (rank,7)
                    assert layer.lora_B['default'].weight.shape == (5,rank)
                    original = reference.base_model.model.blocks[i]
                    assert abs(layer.scaling['default']-original.scaling['default']) < 1e-12
                else:
                    assert type(layer) is nn.Linear
            x = torch.randn(4,7,dtype=torch.float64)
            torch.testing.assert_close(candidate(x),reference(x),rtol=1e-10,atol=1e-11)
        zeros = {k:torch.zeros_like(v) for k,v in tensors.items()}
        output, _, info = compact_zero_padded(zeros,config,[0,0,0])
        assert not output and info['base_only'] and info['compact_elements'] == 0
        corrupted = {k:v.clone() for k,v in tensors.items()}
        corrupted['base_model.model.blocks.1.lora_A.weight'][7].fill_(1)
        corrupted['base_model.model.blocks.1.lora_B.weight'][:,7].fill_(1)
        try:
            compact_zero_padded(corrupted,config,ranks)
        except ValueError as exc:
            assert 'nonzero' in str(exc)
        else:
            raise AssertionError('Compaction silently discarded nonzero coordinates')
    print('PASS: saved/reloaded heterogeneous LoRA + rsLoRA, per-layer alpha, rank-zero removal, all-zero adapter, and nonzero-tail rejection')


if __name__ == '__main__':
    with torch.inference_mode():
        main()
