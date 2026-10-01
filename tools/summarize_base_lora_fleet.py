"""Summarize a base-model LoRA fleet survey into co-residence candidates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def family(base):
    # Diffusers-tagged and original-tagged repos share one transformer architecture.
    return base.removesuffix('-Diffusers')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('survey', type=Path)
    parser.add_argument('--top', type=int, default=4)
    args = parser.parse_args()
    data = json.loads(args.survey.read_text())
    ok = [r for r in data['files'] if r['status'] == 'ok']
    print(f"files={len(data['files'])} ok={len(ok)} errors={Counter(r.get('error_type') for r in data['files'] if r['status'] != 'ok')}")
    by_family = defaultdict(list)
    for record in ok:
        if record['lora_pairs'] > 0:
            by_family[family(record['base'])].append(record)
    rows = []
    for name, records in sorted(by_family.items(), key=lambda kv: -sum(r['lora_bf16_bytes'] for r in kv[1])):
        pure = [r for r in records if r['extra_tensor_count'] == 0]
        print(f"\n## {name}: {len(records)} LoRA files in {len({r['repo'] for r in records})} repos; "
              f"pure A/B {len(pure)}; total LoRA BF16 {sum(r['lora_bf16_bytes'] for r in records)/1e9:.2f} GB")
        groups = defaultdict(list)
        for record in pure:
            groups[record['shape_fingerprint']].append(record)
        for fp, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:args.top]:
            ranks = Counter()
            for m in members:
                ranks.update({int(k): v for k, v in m['rank_counts'].items()})
            gb = sum(m['lora_bf16_bytes'] for m in members)/1e9
            authors = Counter(m['repo'].split('/')[0] for m in members).most_common(4)
            pairs = Counter(m['lora_pairs'] for m in members).most_common(2)
            print(f"  shape={fp} n={len(members)} repos={len({m['repo'] for m in members})} "
                  f"BF16={gb:.2f} GB pairs={pairs} ranks={sorted(ranks.items())[:6]} authors={authors}")
            rows.append(dict(family=name, shape_fingerprint=fp, files=len(members),
                             repos=len({m['repo'] for m in members}), lora_bf16_gb=round(gb, 3),
                             rank_layer_counts=dict(sorted(ranks.items())), authors=authors,
                             members=[(m['repo'], m['filename'], m['downloads'], m['lora_bf16_bytes'])
                                      for m in sorted(members, key=lambda m: -m['downloads'])]))
    out = args.survey.with_name('fleet_candidates.json')
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False)+'\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
