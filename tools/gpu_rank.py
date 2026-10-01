import json, os, sys, time
import torch
from safetensors.torch import load_file
BASE=sys.argv[1]; MERG=sys.argv[2]
def load(root):
    idx=json.load(open(os.path.join(root,"model.safetensors.index.json")))["weight_map"]
    sd={}
    for s in sorted(set(idx.values())): sd.update(load_file(os.path.join(root,s), device="cpu"))
    return sd
dev="cuda"
print("loading...", flush=True); b=load(BASE); m=load(MERG)
mods=sys.argv[3:] or [
 "language_model.model.layers.0.self_attn.q_proj.weight",
 "language_model.model.layers.0.self_attn.v_proj.weight",
 "language_model.lm_head.weight",
 "projector.fc1.weight",
]
for key in mods:
    if key not in b or key not in m: print(key,"missing",flush=True); continue
    d=(m[key].to(dev).float()-b[key].to(dev).float())
    s=torch.linalg.svdvals(d)
    s=torch.sort(s,descending=True).values
    mx=s[0].item(); tot=(s**2).sum().item()
    import numpy as np
    sa=s.cpu().numpy()
    print(f"\n{key} shape={tuple(d.shape)} |D|F={d.norm().item():.3f} max_sv={mx:.4f}", flush=True)
    print("  sv[0:8] :", [round(x,5) for x in s[:8].tolist()], flush=True)
    print("  sv[28:40]:", [round(x,5) for x in s[28:40].tolist()], flush=True)
    print("  sv[60:72]:", [round(x,6) for x in s[60:72].tolist()], flush=True)
    for thr in (1e-1,1e-2,1e-3,1e-4,1e-5):
        print(f"  #sv > {thr}*max: {int((sa>thr*mx).sum())}", flush=True)
    for p in (0.99,0.999,0.9999,0.99999):
        c=np.cumsum(sa**2)/tot; print(f"  rank for {p*100:g}% energy: {int(np.searchsorted(c,p)+1)}", flush=True)
