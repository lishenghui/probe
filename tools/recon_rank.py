#!/usr/bin/env python3
"""Reconstruct a merged OpenVLA-OFT model from base + rank-r SVD of (merged-base)."""
import json, os, shutil, sys, time
import torch
from safetensors.torch import load_file, save_file

BASE = sys.argv[1]
MERG = sys.argv[2]
OUT = sys.argv[3]
RANK = int(sys.argv[4]) if len(sys.argv) > 4 else 64

dev = "cuda"

def load_all(root):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for s in sorted(set(idx.values())):
        sd.update(load_file(os.path.join(root, s), device="cpu"))
    return sd, idx

print(f"loading base {BASE}", flush=True)
b, _ = load_all(BASE)
print(f"loading merged {MERG}", flush=True)
m, midx = load_all(MERG)
print(f"base={len(b)} merged={len(m)}", flush=True)

os.makedirs(OUT, exist_ok=True)

# non-weight files
for fn in os.listdir(MERG):
    if fn.startswith("model") and fn.endswith((".safetensors", ".index.json")):
        continue
    src = os.path.join(MERG, fn)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(OUT, fn))

# reconstruct
out = {}
n_rec = 0
import numpy as np
errs = []
t0 = time.time()
for i, (k, v) in enumerate(m.items()):
    if k in b and tuple(b[k].shape) == tuple(v.shape) and v.dim() == 2:
        d = v.to(dev).float() - b[k].to(dev).float()
        if torch.count_nonzero(d) == 0:
            out[k] = v
        else:
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            r = min(RANK, S.numel())
            rec = (U[:, :r] * S[:r]) @ Vh[:r, :]
            W = b[k].to(dev).float() + rec
            out[k] = W.to(v.dtype).cpu()
            rel = (rec - d).norm().item() / (d.norm().item() + 1e-12)
            errs.append(rel)
            n_rec += 1
    else:
        out[k] = v
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(m)}  reconstructed={n_rec}  t={time.time()-t0:.0f}s", flush=True)

print(f"reconstructed {n_rec} modules at rank {RANK}; mean rel res={np.mean(errs):.2e} max={np.max(errs):.2e}", flush=True)

# save shards per merged index
by_shard = {}
for k in out:
    by_shard.setdefault(midx[k], {})[k] = out[k]
for shard, d in sorted(by_shard.items()):
    p = os.path.join(OUT, shard)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    save_file(d, p)
    print(f"  wrote {shard} ({len(d)} tensors)", flush=True)
print("DONE", OUT, flush=True)
