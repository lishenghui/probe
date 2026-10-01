"""Download one fleet chosen by a fleet survey, verify LFS sha256, and write a manifest.

Selection: either ``--manifest`` (re-fetch the exact pinned files of an existing
fleet manifest, e.g. on another machine) or every surveyed file whose repo starts with ``--repo-prefix`` and whose
base family and shape fingerprint match. Each adapter's model card is saved too,
because trigger phrases and example prompts live there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from survey_base_lora_fleet import api, with_backoff  # noqa: E402


def fetch(url, destination):
    request = urllib.request.Request(url, headers={'User-Agent': 'PROBE-fleet-fetch/1.0'})
    partial = destination.with_suffix(destination.suffix + '.part')
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as handle:
        while chunk := response.read(8 << 20):
            digest.update(chunk)
            handle.write(chunk)
    partial.replace(destination)
    return digest.hexdigest()


def trigger_phrase(card):
    # Remade-AI cards: "The key trigger phrase is: <code ...>phrase</code>"
    match = re.search(r'trigger phrase is:\s*<code[^>]*>(.*?)</code>', card, re.S)
    return match.group(1).strip() if match else None


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def refetch(manifest, output_dir):
    for index, entry in enumerate(manifest['entries'], 1):
        target = output_dir / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        base_url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/"
        if not (target.exists() and sha256_of(target) == entry['sha256']):
            if with_backoff(fetch, base_url + urllib.parse.quote(entry['filename']), target) != entry['sha256']:
                raise ValueError(f"sha256 mismatch for {entry['repo']}")
        with_backoff(fetch, base_url + 'README.md', target.parent / 'README.md')
        print(f"[{index}/{len(manifest['entries'])}] {entry['repo']} ok", flush=True)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='re-fetch exactly the pinned entries of this manifest')
    parser.add_argument('--survey', type=Path)
    parser.add_argument('--base-family')
    parser.add_argument('--shape-fingerprint')
    parser.add_argument('--repo-prefix')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.manifest:
        return refetch(json.loads(args.manifest.read_text()), args.output_dir)
    survey = json.loads(args.survey.read_text())
    chosen = [r for r in survey['files'] if r['status'] == 'ok'
              and r['base'].removesuffix('-Diffusers') == args.base_family
              and r['shape_fingerprint'] == args.shape_fingerprint
              and r['repo'].startswith(args.repo_prefix)]
    chosen.sort(key=lambda r: -r['downloads'])
    print(f'{len(chosen)} adapters, {sum(r["file_bytes"] for r in chosen)/1e9:.2f} GB', flush=True)
    entries = []
    for index, record in enumerate(chosen, 1):
        repo, filename = record['repo'], record['filename']
        info = api(f'models/{repo}/revision/main?blobs=true')
        sibling = next(s for s in info['siblings'] if s['rfilename'] == filename)
        expected = sibling['lfs']['sha256']
        folder = args.output_dir / 'adapters' / repo.split('/', 1)[1]
        folder.mkdir(parents=True, exist_ok=True)
        weights = folder / Path(filename).name
        base_url = f'https://huggingface.co/{repo}/resolve/{info["sha"]}/'
        if weights.exists() and weights.stat().st_size == record['file_bytes']:
            actual = sha256_of(weights)
        else:
            start = time.time()
            actual = with_backoff(fetch, base_url + urllib.parse.quote(filename), weights)
            print(f'[{index}/{len(chosen)}] {repo} {record["file_bytes"]/1e6:.0f} MB '
                  f'in {time.time()-start:.0f}s', flush=True)
        if actual != expected:
            raise ValueError(f'sha256 mismatch for {repo}/{filename}')
        if any(s['rfilename'] == 'README.md' for s in info['siblings']):
            with_backoff(fetch, base_url + 'README.md', folder / 'README.md')
        card = folder / 'README.md'
        entries.append(dict(
            trigger=trigger_phrase(card.read_text()) if card.exists() else None,
            name=repo.split('/', 1)[1], repo=repo, revision=info['sha'], filename=filename,
            path=str(weights.relative_to(args.output_dir)), sha256=actual,
            file_bytes=record['file_bytes'], lora_parameters=record['lora_parameters'],
            lora_pairs=record['lora_pairs'], rank_counts=record['rank_counts'],
            alpha_tensors=record['alpha_tensors'], dtype_parameters=record['dtype_parameters'],
            downloads_at_survey=record['downloads']))
    manifest = dict(
        fleet=f'{args.repo_prefix.rstrip("/")} on {args.base_family}', base_model=args.base_family,
        shape_fingerprint=args.shape_fingerprint, survey=str(args.survey),
        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), host=os.uname().nodename,
        adapters=len(entries), total_file_bytes=sum(e['file_bytes'] for e in entries),
        total_lora_parameters=sum(e['lora_parameters'] for e in entries), entries=entries)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'wrote {args.output_dir/"manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
