"""Belebele accuracy of Qwen3-8B with and without per-language LoRA adapters.

Two scorers, because the adapters' training prompt format is not published:
  * letter: chat-templated multiple-choice prompt (thinking disabled), argmax of the
    next-token logits over " A"/" B"/" C"/" D" variants;
  * option: mean token log-likelihood of each option text as the assistant reply.
All 900 questions per language are scored. The adapters were trained on an
unpublished ~190-question subset of Belebele, so part of the set may be seen data;
results are reported as-is and must not be read as held-out accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = 'ABCD'


def prompt_messages(row):
    options = '\n'.join(f"{LETTERS[i]}. {row[f'mc_answer{i+1}']}" for i in range(4))
    text = (f"{row['flores_passage']}\n\nQuestion: {row['question']}\n{options}\n\n"
            "Answer with the letter of the correct option.")
    return [{'role': 'user', 'content': text}]


@torch.no_grad()
def score_letters(model, tok, rows, batch):
    letter_ids = [[tok.encode(v, add_special_tokens=False)[0] for v in (l, f' {l}')] for l in LETTERS]
    correct = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        texts = [tok.apply_chat_template(prompt_messages(r), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for r in chunk]
        enc = tok(texts, return_tensors='pt', padding=True).to(model.device)
        logits = model(**enc).logits[:, -1].float()
        scores = torch.stack([logits[:, ids].max(dim=1).values for ids in letter_ids], dim=1)
        pred = scores.argmax(dim=1).tolist()
        correct += sum(int(p == int(r['correct_answer_num']) - 1) for p, r in zip(pred, chunk))
    return correct / len(rows)


@torch.no_grad()
def score_options(model, tok, rows, batch):
    correct = 0
    for row in rows:
        prefix = tok.apply_chat_template(prompt_messages(row), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        prefix_len = len(tok(prefix, add_special_tokens=False).input_ids)
        texts = [prefix + row[f'mc_answer{i+1}'] for i in range(4)]
        enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        logp = model(**enc).logits[:, :-1].float().log_softmax(-1)
        target = enc.input_ids[:, 1:]
        token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mask = enc.attention_mask[:, 1:].clone()
        pad_left = (enc.attention_mask == 0).sum(dim=1)
        for j in range(4):  # score only the option tokens (left padding shifts the prefix)
            mask[j, :pad_left[j] + prefix_len - 1] = 0
        mean = (token_logp * mask).sum(1) / mask.sum(1).clamp_min(1)
        correct += int(mean.argmax().item() == int(row['correct_answer_num']) - 1)
    return correct / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True, help='dir with <lang-code>/adapter_model.safetensors')
    parser.add_argument('--data', type=Path, required=True, help='dir with <Lang_Script>.jsonl')
    parser.add_argument('--languages', nargs='+', default=['afr_Latn', 'swh_Latn', 'hin_Deva', 'arb_Arab', 'jpn_Jpan'])
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='cuda').eval()
    results = dict(model=str(args.model), adapters=str(args.adapters), scorers=['letter', 'option'], rows=[])
    model = None
    for lang in args.languages:
        rows = [json.loads(l) for l in (args.data / f'{lang}.jsonl').read_text().splitlines() if l.strip()]
        rows = rows[:args.limit] if args.limit else rows
        adapter_dir = args.adapters / lang.lower().replace('_', '-')
        for label in ('base', 'lora'):
            start = time.perf_counter()
            if label == 'base':
                net = base
            else:
                net = PeftModel.from_pretrained(base, adapter_dir, adapter_name=lang).eval()
            row = dict(language=lang, model=label, n=len(rows),
                       letter_acc=score_letters(net, tok, rows, args.batch),
                       option_acc=score_options(net, tok, rows, args.batch))
            if label == 'lora':
                row['card_eval_mcq_accuracy'] = json.loads((adapter_dir / 'eval_results.json').read_text()).get(
                    'eval_mcq_accuracy')
                base = net.unload()  # remove LoRA layers, restoring the plain base model
            row['seconds'] = time.perf_counter() - start
            results['rows'].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(results, indent=1))
    print('done', flush=True)


if __name__ == '__main__':
    main()
