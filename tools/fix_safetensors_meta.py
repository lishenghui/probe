import os, sys, glob, json
from safetensors.torch import load_file, save_file
D=sys.argv[1]
for p in sorted(glob.glob(os.path.join(D,"model-*.safetensors"))):
    d=load_file(p)
    tmp=p+".tmp"
    save_file(d, tmp, metadata={"format":"pt"})
    os.replace(tmp, p)
    print("fixed", os.path.basename(p), flush=True)
print("DONE", flush=True)
