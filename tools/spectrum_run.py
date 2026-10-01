import json, struct, os, sys, time
import numpy as np
BASE="/nobackup/proj/disk/bloom/personal/shenghui/probe/artifacts/rq3/models/openvla-7b-base"
MERG="/nobackup/proj/disk/bloom/personal/shenghui/probe/artifacts/rq3/models/rlinf-libero130-rl"
def imap(d): return json.load(open(os.path.join(d,"model.safetensors.index.json")))["weight_map"]
def rt(root, wm, key):
    p=os.path.join(root,wm[key])
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); info=h[key]
        st,en=info['data_offsets']; f.seek(8+n+st); raw=f.read(en-st)
    dt=info['dtype']
    if dt=='BF16':
        arr=(np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
    else:
        arr=np.frombuffer(raw,dtype={'F32':np.float32,'F16':np.float16}[dt]).astype(np.float32)
    return arr.reshape(info['shape'])
bm,mm=imap(BASE),imap(MERG)
print("maps loaded", flush=True)
mods=sys.argv[1:]
for key in mods:
    t0=time.time()
    if key not in bm or key not in mm:
        print(f"{key} MISSING base={key in bm} merged={key in mm}", flush=True); continue
    b=rt(BASE,bm,key); m=rt(MERG,mm,key)
    D=(m-b).astype(np.float32)
    s=np.linalg.svd(D, compute_uv=False)
    s=np.sort(s)[::-1]; tot=float((s**2).sum())
    nz=int((s>s[0]*1e-4).sum())
    print(f"{key} shape={D.shape} |D|F={np.sqrt(tot):.4f} rank>1e-4max={nz} t={time.time()-t0:.1f}s", flush=True)
    print("  top12:", np.round(s[:12],6).tolist(), flush=True)
