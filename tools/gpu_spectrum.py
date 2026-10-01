import json, os, sys, time
import torch
from safetensors.torch import load_file

BASE=sys.argv[1]; MERG=sys.argv[2]
def load(root):
    idx=json.load(open(os.path.join(root,"model.safetensors.index.json")))["weight_map"]
    shards=sorted(set(idx.values()))
    sd={}
    for s in shards:
        sd.update(load_file(os.path.join(root,s), device="cpu"))
    return sd, idx
dev="cuda"
print("loading base...", flush=True); b,_=load(BASE)
print("loading merged...", flush=True); m,_=load(MERG)
print(f"base tensors={len(b)} merged tensors={len(m)}", flush=True)

keys=[k for k in m if k in b and m[k].shape==b[k].shape and m[k].dim()==2]
print(f"common 2D tensors: {len(keys)}", flush=True)

def spec(key):
    d=(m[key].to(dev).float()-b[key].to(dev).float())
    s=torch.linalg.svdvals(d)
    s,_srt=torch.sort(s, descending=True); s=s[:64].tolist()
    return d, s

mods=sys.argv[3:] or [
 "language_model.model.layers.0.self_attn.q_proj.weight",
 "language_model.model.layers.0.self_attn.k_proj.weight",
 "language_model.model.layers.0.self_attn.v_proj.weight",
 "language_model.model.layers.0.self_attn.o_proj.weight",
 "language_model.model.layers.0.mlp.down_proj.weight",
 "language_model.model.layers.0.mlp.gate_proj.weight",
 "language_model.model.layers.0.mlp.up_proj.weight",
 "language_model.lm_head.weight",
 "projector.fc1.weight",
 "vision_backbone.featurizer.blocks.0.attn.qkv.weight",
]
for key in mods:
    if key not in b or key not in m:
        print(f"{key}: missing base={key in b} merged={key in m}", flush=True); continue
    t=time.time(); d,s=spec(key)
    import numpy as np
    sarr=np.array(s); tot=float((sarr**2).sum())
    nz=int((sarr > sarr[0]*1e-4).sum())
    # full spectrum count via svdvals count
    print(f"{key} shape={tuple(d.shape)} |D|F={d.norm().item():.4f} rank(sv>1e-4max)={nz} t={time.time()-t:.1f}s", flush=True)
    print("  top12:", [round(x,6) for x in s[:12]], flush=True)
    del d
