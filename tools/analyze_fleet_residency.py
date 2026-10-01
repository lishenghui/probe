"""Turn measured fleet residency into capacity and cache-latency results.

Inputs are all measured by bench_fleet_residency.py: the no-adapter generation peak,
per-adapter resident bytes, miss reload time and generation time per variant.
Derived here:
  * capacity N(B): adapters that stay resident within a GPU budget B;
  * whether the whole fleet plus the pipeline fits in B;
  * an LRU adapter cache replayed on a Zipf request stream over the fleet, with the
    measured reload cost per miss, giving hit rate and mean per-request latency.
Generation time can be overridden (--gen-seconds) to show the few-step regime,
labelled as such; it is never mixed with the measured 30-step numbers.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import random


def lru_replay(requests, capacity, reload_s, gen_s):
    cache, misses = OrderedDict(), 0
    for adapter in requests:
        if adapter in cache:
            cache.move_to_end(adapter)
        else:
            misses += 1
            if capacity == 0:
                continue
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[adapter] = True
    hit = 1 - misses / len(requests)
    return hit, gen_s + (1 - hit) * reload_s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('residency', type=Path)
    parser.add_argument('--budgets-gib', type=float, nargs='+', default=[24, 32, 48, 64, 80, 96])
    parser.add_argument('--zipf', type=float, default=1.0)
    parser.add_argument('--requests', type=int, default=20000)
    parser.add_argument('--gen-seconds', type=float, nargs='*', default=[],
                        help='extra hypothetical generation times (e.g. few-step models)')
    args = parser.parse_args()
    data = json.loads(args.residency.read_text())
    floor = data['no_adapter']['peak_bytes']
    variants = data['variants']
    n = next(iter(variants.values()))['adapters']

    rng = random.Random(0)
    weights = [1 / (i + 1)**args.zipf for i in range(n)]
    requests = rng.choices(range(n), weights=weights, k=args.requests)

    out = dict(source=str(args.residency), no_adapter_peak_gib=floor / 2**30, zipf=args.zipf, capacity={}, cache={})
    print(f"no-adapter generation peak {floor/2**30:.2f} GiB on {data['gpu']}")
    print(f"{'variant':9s} {'fleet GiB':>9s} {'per-ad MB':>9s} {'peak w/ fleet':>13s} " +
          ' '.join(f'N({b:g}G)' for b in args.budgets_gib))
    for name, v in variants.items():
        per = v['per_adapter_resident_bytes']
        caps = {b: max(0, min(n, int((b * 2**30 - floor) // per))) for b in args.budgets_gib}
        out['capacity'][name] = dict(per_adapter_resident_mb=per / 1e6, fleet_resident_gib=v['fleet_resident_bytes'] / 2**30,
                                     measured_peak_with_fleet_gib=v['generation_peak_bytes'] / 2**30,
                                     adapters_resident=caps,
                                     whole_fleet_fits={b: v['generation_peak_bytes'] <= b * 2**30 for b in args.budgets_gib})
        print(f"{name:9s} {v['fleet_resident_bytes']/2**30:9.2f} {per/1e6:9.0f} {v['generation_peak_bytes']/2**30:13.2f} " +
              ' '.join(f'{caps[b]:7d}' for b in args.budgets_gib))

    regimes = [('measured', None)] + [(f'assumed_{g:g}s', g) for g in args.gen_seconds]
    for label, gen in regimes:
        out['cache'][label] = {}
        print(f"\nLRU cache, {label} generation time; mean latency per request (s) / hit rate")
        for name, v in variants.items():
            per = v['per_adapter_resident_bytes']
            g = v['generation_s'] if gen is None else gen
            row = {}
            for b in args.budgets_gib:
                cap = max(0, min(n, int((b * 2**30 - floor) // per)))
                hit, latency = lru_replay(requests, cap, v['miss_reload_from_disk_s'], g)
                row[b] = dict(capacity=cap, hit_rate=hit, mean_latency_s=latency)
            out['cache'][label][name] = row
            print(f"{name:9s} " + ' '.join(f"{row[b]['mean_latency_s']:7.1f}/{row[b]['hit_rate']:.2f}" for b in args.budgets_gib))
    target = args.residency.with_name('residency_analysis.json')
    target.write_text(json.dumps(out, indent=1))
    print(f'\nwrote {target}')


if __name__ == '__main__':
    main()
