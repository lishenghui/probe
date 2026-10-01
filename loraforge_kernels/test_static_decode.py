"""Check captured KV-cache progression against ordinary greedy generation."""

import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from .peft_integration import enable_loraforge_peft
from .static_decode import StaticDecodeRunner


@torch.inference_mode()
def main():
    torch.manual_seed(53)
    os.environ['LORAFORGE_TUNE_MODE'] = 'graph'
    cfg = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512,
                      bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = LlamaForCausalLM(cfg).cuda().bfloat16().eval()
    for adapted in (False, True):
        if adapted:
            model = get_peft_model(model, LoraConfig(r=3, lora_alpha=3,
                target_modules=['q_proj', 'v_proj'])).eval()
            for module in model.modules():
                if hasattr(module, 'lora_B'):
                    module.lora_B['default'].weight.normal_(std=0.03)
            enable_loraforge_peft(model, enable_grouping=False)
        for batch in (1, 2):
            ids = torch.randint(3, 64, (batch, 7), device='cuda')
            for count in (1, 6):
                runner = StaticDecodeRunner(model, ids, count)
                for _ in range(2):
                    ids.random_(3, 64)
                    want = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        do_sample=False, min_new_tokens=count, max_new_tokens=count, pad_token_id=0)
                    eager = runner(ids, replay=False)
                    got = runner(ids)
                    assert torch.equal(got, eager), (adapted, batch, count, 'graph/eager')
                    assert torch.equal(got, want), (adapted, batch, count, 'generate')
    # Packed prefill rebinds base weights; graph-captured decode must retain
    # its original storage across later prefill requests and allocator reuse.
    os.environ['LORAFORGE_VARIANT'] = 'concat'
    ids = torch.randint(3, 64, (2, 129), device='cuda')
    runner = StaticDecodeRunner(model, ids, 6)
    for _ in range(3):
        ids.random_(3, 64)
        assert runner.validate(ids) == 0.0
        assert torch.equal(runner(ids), runner(ids, replay=False))
    os.environ.pop('LORAFORGE_VARIANT')
    print('PASS: static cache/position/mask/argmax replay matches generate; base and nonzero LoRA; changing prompts')


if __name__ == '__main__':
    main()
