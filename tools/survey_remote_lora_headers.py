"""Read bounded public safetensors headers; never download tensor payloads."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import urllib.parse
import urllib.request


CANDIDATES = [
    ('LTX-2 distilled', 'Lightricks/LTX-2', 'ltx-2-19b-distilled-lora-384.safetensors'),
    ('Wan Pusa', 'Kijai/WanVideo_comfy', 'Pusa/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors'),
    ('Wan LightX2V', 'Kijai/WanVideo_comfy', 'Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors'),
    ('HunyuanVideo AnimeShots', 'trojblue/HunyuanVideo-lora-AnimeShots', 'v0.1/adapter_model.safetensors'),
    ('FLUX Krea BLAZE', 'MintLab/FLUX-Krea-BLAZE', 'LORA/Flux_Krea_Blaze_Lora-rank128.safetensors'),
    ('Hyper-FLUX 8 steps', 'ByteDance/Hyper-SD', 'Hyper-FLUX.1-dev-8steps-lora.safetensors'),
    ('Wan identity example', 'malcolmrey/wan', 'wan2.1/wan_amandapeet_v1.safetensors'),
]


def bounded_range(url, start, end):
    # Separate URLs avoid intermediaries reusing a different Range response.
    query = urllib.parse.urlencode({'header_probe': f'{start}-{end}'})
    request = urllib.request.Request(f'{url}?{query}', headers={
        'Range': f'bytes={start}-{end}', 'User-Agent': 'PROBE-header-survey/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206:
            raise ValueError('Server ignored Range; refusing full-file download')
        content_range = response.headers.get('Content-Range', '')
        if not content_range.startswith(f'bytes {start}-{end}/'):
            raise ValueError(f'Unexpected Content-Range: {content_range}')
        data = response.read(end-start+2)
        if len(data) != end-start+1:
            raise ValueError('Incorrect range length')
        return data, int(content_range.split('/')[-1])


def probe(candidate):
    label, repo, filename = candidate
    blob = f'https://huggingface.co/{repo}/blob/main/{urllib.parse.quote(filename)}'
    resolve = f'https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(filename)}'
    record = dict(label=label, repo=repo, filename=filename, source=blob)
    try:
        prefix, total = bounded_range(resolve, 0, 7)
        length = struct.unpack('<Q', prefix)[0]
        if not 0 < length <= 4*1024*1024:
            raise ValueError(f'Header exceeds 4 MiB limit: {length}')
        raw, again = bounded_range(resolve, 8, 7+length)
        assert total == again
        header = json.loads(raw)
        tensors = {k:v for k,v in header.items() if k != '__metadata__'}
        parameters = sum(math.prod(v['shape']) for v in tensors.values())
        payload = sum(v['data_offsets'][1]-v['data_offsets'][0] for v in tensors.values())
        dtype_params = Counter()
        ranks = Counter()
        unpaired = []
        extras = []
        pairs = 0
        for key, value in tensors.items():
            dtype_params[value['dtype']] += math.prod(value['shape'])
            if key.endswith(('.lora_A.weight', '.lora_down.weight')):
                rank = value['shape'][0]
                ranks[rank] += 1
                other = key.replace('.lora_A.weight', '.lora_B.weight').replace('.lora_down.weight', '.lora_up.weight')
                if other not in tensors or tensors[other]['shape'][1] != rank:
                    unpaired.append(key)
                else:
                    pairs += 1
            elif not key.endswith(('.lora_B.weight', '.lora_up.weight')):
                extras.append(key)
        record.update(status='ok', file_bytes=total, payload_bytes=payload,
                      downloaded_header_bytes=length+8, header_sha256=hashlib.sha256(raw).hexdigest(),
                      tensor_count=len(tensors), tensor_parameters=parameters,
                      dtype_parameters=dict(dtype_params), rank_counts=dict(ranks),
                      lora_pairs=pairs, unpaired_a_keys=unpaired,
                      extra_tensor_count=len(extras), extra_tensor_sample=extras[:12],
                      all_tensors_bf16_bytes=parameters*2,
                      metadata=header.get('__metadata__', {}))
    except Exception as exc:
        # Never emit redirect/signed URLs or request headers.
        record.update(status='error', error_type=type(exc).__name__)
    print(json.dumps({k:v for k,v in record.items() if k not in ('metadata','source')}, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a prior survey')
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(probe, CANDIDATES))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        job_id=os.environ.get('SLURM_JOB_ID'),
        scope='Public file metadata only: HTTP byte ranges for safetensors headers; no tensor payloads',
        candidates=records), indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
