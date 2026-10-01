"""Where can adapter compression turn into end-to-end gains? A bytes-on-the-critical-path map.

rho_active = (distinct adapters touched per step x bytes per adapter) / base weight bytes.
Bandwidth-bound step (LLM decode): time ~ base + KV + active adapter bytes, so saving a
fraction s of adapter bytes gives speedup 1 / (1 - s*rho/(1 + kv + rho)).
Compute-bound step (video DiT, LLM prefill): adapter FLOPs ~ 2*r*(d_in+d_out) per token vs
~2*params for the base, i.e. the adapter-parameter fraction; savings are bounded by it.
Scenario sizes are from measured files/audits in this repo unless marked 'assumed'.
"""
import json

GB = 1e9
rows = [
    # name, base bytes, bytes per adapter, adapters resident, distinct active per step, bound, source
    ('Wan2.1-I2V-14B video, BF16 base', 32.8*GB, 0.359*GB, 49, 1, 'compute', 'measured (Remade fleet)'),
    ('Wan2.1-I2V-14B video, FP8 base', 16.4*GB, 0.359*GB, 49, 1, 'compute', 'measured sizes'),
    ('LoRA Land on 7B, rank 8', 14.5*GB, 78/12*2**20, 12, 1, 'bandwidth', 'audit 3222950'),
    ('Lots-of-LoRAs on 7B, rank 16', 14.5*GB, 450/25*2**20, 25, 1, 'bandwidth', 'audit 3222950'),
    ('LoRARetriever on 7B, rank 8', 14.5*GB, 328/41*2**20, 41, 1, 'bandwidth', 'audit 3222950'),
    ('LLM 7B, rank 64, batch of 8 distinct adapters', 14.5*GB, 0.32*GB, 1000, 8, 'bandwidth', 'assumed (S-LoRA-like)'),
    ('LLM 7B, rank 64, batch of 32 distinct adapters', 14.5*GB, 0.32*GB, 1000, 32, 'bandwidth', 'assumed (S-LoRA-like)'),
    ('LLM 1.5B, rank 64, batch of 32 distinct adapters', 3.1*GB, 0.08*GB, 1000, 32, 'bandwidth', 'assumed'),
]
s, kv = 0.5, 0.1  # b50 saves ~half the adapter bytes; KV traffic assumed 10% of base weights
out = []
print(f"{'scenario':50s} {'fleet/base':>10s} {'rho_active':>10s} {'step speedup @b50':>17s}")
for name, base, per, n, active, bound, src in rows:
    rho_fleet, rho = n*per/base, active*per/base
    if bound == 'bandwidth':
        speed = 1/(1 - s*rho/(1 + kv + rho))
    else:
        speed = 1/(1 - s*rho/(1 + rho))  # FLOP share of the adapter, an optimistic upper bound
    out.append(dict(scenario=name, bound=bound, source=src, fleet_over_base=rho_fleet, rho_active=rho,
                    predicted_step_speedup_b50=speed))
    print(f"{name:50s} {rho_fleet:10.3f} {rho:10.4f} {speed:17.3f}x  [{bound}; {src}]")
json.dump(dict(assumptions=dict(saved_fraction=s, kv_over_base=kv), rows=out),
          open('docs/benefit_map/benefit_map.json', 'w'), indent=1)
