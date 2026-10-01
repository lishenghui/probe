import json, struct, os, sys
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
for key in sys.argv[1:]:
    if key not in bm or key not in mm: print(key,'missing'); continue
    b=rt(BASE,bm,key); m=rt(MERG,mm,key)
    D=(m-b).astype(np.float32)
    k=min(80,min(D.shape))
    from scipy.sparse.linalg import svds
    try:
        s=np.linalg.svd(D, compute_uv=False)  # full for correctness if small enough
    except Exception:
        s=svds(D,k=k,return_singular_vectors=False)[::-1]
    s=np.sort(s)[::-1]; tot=(s**2).sum()
    print(f"\n{key} shape={D.shape} |D|F={np.sqrt(tot):.4f}")
    print("  top12 sv:", np.round(s[:12],6))
    nz=int((s>s[0]*1e-4).sum())
    print("  count sv > 1e-4*max:", nz)
