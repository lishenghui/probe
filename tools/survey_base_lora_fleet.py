"""Survey every public LoRA tagged with one base model; read safetensors headers only.

For each base model, list Hugging Face repos tagged ``base_model:adapter:<base>``,
then read the bounded JSON header of each ``.safetensors`` file. No tensor payload
is downloaded. The output groups adapters by a layer-shape fingerprint so that a
co-resident fleet (one base, many adapters) can be chosen from adapters whose
A/B factors attach to the same linear layers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from survey_remote_lora_headers import bounded_range  # noqa: E402

PAIRS = (('.lora_A.weight', '.lora_B.weight'), ('.lora_down.weight', '.lora_up.weight'),
         ('.lora.down.weight', '.lora.up.weight'))
PREFIXES = ('base_model.model.', 'diffusion_model.', 'transformer.', 'model.diffusion_model.')
MAX_HEADER = 8*1024*1024


def with_backoff(call, *args, attempts=6):
    # Anonymous Hub access is rate limited; honour Retry-After instead of failing.
    for attempt in range(attempts):
        try:
            return call(*args)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts-1:
                raise
            time.sleep(min(300, int(exc.headers.get('Retry-After') or 30*(attempt+1))))


def _api(path):
    request = urllib.request.Request(f'https://huggingface.co/api/{path}',
                                     headers={'User-Agent': 'PROBE-fleet-survey/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def api(path):
    return with_backoff(_api, path)


def canonical(module):
    for prefix in PREFIXES:
        if module.startswith(prefix):
            module = module[len(prefix):]
    # kohya keys flatten dots to underscores; keep them distinct but comparable
    return re.sub(r'^lora_unet_', '', module)


def probe(job):
    base, repo, downloads, filename, max_bytes = job
    record = dict(base=base, repo=repo, filename=filename, downloads=downloads)
    resolve = f'https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(filename)}'
    try:
        prefix, total = with_backoff(bounded_range, resolve, 0, 7)
        if total > max_bytes:
            raise ValueError(f'file of {total} bytes exceeds adapter size limit')
        length = struct.unpack('<Q', prefix)[0]
        if not 0 < length <= MAX_HEADER:
            raise ValueError(f'header length {length}')
        raw, _ = with_backoff(bounded_range, resolve, 8, 7+length)
        header = json.loads(raw)
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        ranks, dtypes, shapes, modules = Counter(), Counter(), Counter(), []
        extras, alphas, lora_params = [], 0, 0
        paired = set()
        for key, value in tensors.items():
            dtypes[value['dtype']] += math.prod(value['shape'])
            for a_suffix, b_suffix in PAIRS:
                if key.endswith(a_suffix):
                    other = key[:-len(a_suffix)] + b_suffix
                    if other in tensors and len(value['shape']) == 2:
                        r, d_in = value['shape']
                        d_out = tensors[other]['shape'][0]
                        ranks[r] += 1
                        shapes[f'{d_out}x{d_in}'] += 1
                        lora_params += r*(d_in+d_out)
                        modules.append(canonical(key[:-len(a_suffix)]))
                        paired.update((key, other))
                    break
        for key in tensors:
            if key in paired:
                continue
            if key.endswith('.alpha'):
                alphas += 1
            else:
                extras.append(key)
        params = sum(math.prod(v['shape']) for v in tensors.values())
        fingerprint = hashlib.sha256(json.dumps(sorted(shapes.items())).encode()).hexdigest()[:12]
        record.update(
            status='ok', file_bytes=total, tensor_parameters=params, lora_parameters=lora_params,
            lora_bf16_bytes=2*lora_params, dtype_parameters=dict(dtypes),
            rank_counts={str(k): v for k, v in ranks.items()}, lora_pairs=len(modules),
            alpha_tensors=alphas, extra_tensor_count=len(extras), extra_tensor_sample=extras[:6],
            shape_counts=dict(shapes), shape_fingerprint=fingerprint,
            module_fingerprint=hashlib.sha256('\n'.join(sorted(modules)).encode()).hexdigest()[:12],
            module_sample=sorted(modules)[:3])
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        record.update(status='error', error_type=type(exc).__name__, error=str(exc)[:160])
    return record


def jobs_for(base, limit_repos, max_file_bytes):
    # One listing call returns file names too; per-repo info calls triggered HTTP 429.
    query = urllib.parse.urlencode({'filter': f'base_model:adapter:{base}', 'limit': 1000,
                                    'sort': 'downloads', 'direction': -1})
    repos = api(f'models?{query}&expand[]=downloads&expand[]=gated&expand[]=siblings')[:limit_repos]
    jobs, skipped = [], []
    for model in repos:
        if model.get('gated'):
            skipped.append((model['id'], 'gated'))
            continue
        for sibling in model.get('siblings') or []:
            if sibling['rfilename'].endswith('.safetensors'):
                jobs.append((base, model['id'], model.get('downloads', 0), sibling['rfilename'], max_file_bytes))
    return jobs, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-repos', type=int, default=1000)
    parser.add_argument('--max-file-gb', type=float, default=12.0,
                        help='skip files larger than this (full checkpoints, not adapters)')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a prior survey')
    jobs, skipped = [], []
    for base in args.base:
        found, missed = jobs_for(base, args.limit_repos, int(args.max_file_gb*1e9))
        print(f'{base}: {len(found)} safetensors files, {len(missed)} repos skipped', flush=True)
        jobs += found
        skipped += [(base, *item) for item in missed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(probe, jobs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        job_id=os.environ.get('SLURM_JOB_ID'), bases=args.base,
        scope='HF API listings plus safetensors header byte ranges; no tensor payloads',
        skipped=skipped, files=records), indent=1, ensure_ascii=False)+'\n')
    print(f'wrote {len(records)} records, {sum(r["status"] == "ok" for r in records)} ok', flush=True)


if __name__ == '__main__':
    main()
