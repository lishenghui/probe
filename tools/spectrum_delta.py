#!/usr/bin/env python3
"""Measure the effective rank of (merged - base) per module via SVD."""
import json, struct, sys, os
import numpy as np

BASE = "/nobackup/proj/disk/bloom/personal/shenghui/probe/artifacts/rq3/models/openvla-7b-base"
MERG = "/nobackup/proj/disk/bloom/personal/shenghui/probe/artifacts/rq3/models/rlinf-libero130-rl"

_DTYPE = {"F32": np.float32, "F16": np.float16, "BF16": None, "I64": np.int64, "I32": np.int32, "U8": np.uint8}

def index_map(d):
    p = os.path.join(d, "model.safetensors.index.json")
    return json.load(open(p))["weight_map"]

def read_tensor(root, weight_map, key):
    p = os.path.join(root, weight_map[key])
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        info = h[key]
        start, end = info["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    dt = info["dtype"]
    if dt == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        arr = u.view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=_DTYPE[dt]).astype(np.float32)
    return arr.reshape(info["shape"])

def main(modules):
    bm, mm = index_map(BASE), index_map(MERG)
    for key in modules:
        if key not in bm or key not in mm:
            print(f"{key}: base={key in bm} merged={key in mm}  SKIP"); continue
        b = read_tensor(BASE, bm, key); m = read_tensor(MERG, mm, key)
        if b.shape != m.shape:
            print(f"{key}: shape {b.shape} vs {m.shape} SKIP"); continue
        d = (m - b).astype(np.float64)
        s = np.linalg.svd(d, compute_uv=False)
        tot = (s ** 2).sum()
        if tot == 0:
            print(f"{key}: delta is all zeros"); continue
        cum = np.cumsum(s ** 2) / tot
        def rank_at(p): return int(np.searchsorted(cum, p) + 1)
        nz = int((s > s[0] * 1e-4).sum())
        print(f"\n{key}  shape={b.shape}  |delta|_F={np.sqrt(tot):.4f}")
        print(f"  top10 sv: {np.round(s[:10], 5)}")
        print(f"  rank>1e-4*max: {nz}")
        for p in (0.9, 0.99, 0.999, 0.9999, 1.0 - 1e-12):
            print(f"  rank for {p*100:g}% energy: {rank_at(p)}")

if __name__ == "__main__":
    mods = sys.argv[1:] or [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
    ]
    main(mods)
