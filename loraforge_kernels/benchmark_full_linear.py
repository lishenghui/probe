#!/usr/bin/env python3
"""End-to-end base Linear + LoRA benchmark for the epilogue-fusion boundary."""
import json, statistics
from pathlib import Path
import torch
from loraforge_kernels import fused_lora

def bench(fn, warmup=40, repeats=300):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); xs=[]
    for _ in range(repeats):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize()
        xs.append(a.elapsed_time(b)*1000)
    return {"median_us":statistics.median(xs),"mean_us":statistics.fmean(xs),"std_us":statistics.stdev(xs)}

def main():
    torch.manual_seed(1); d=torch.float16; dev="cuda"; k=n=4096; rows=[]
    for m in (1,4,8):
      x=torch.randn(m,k,device=dev,dtype=d); w=torch.randn(n,k,device=dev,dtype=d)/k**.5
      base=bench(lambda:x@w.T)
      for r in (1,4,8,16,32):
        a=torch.randn(r,k,device=dev,dtype=d)/k**.5; b=torch.randn(n,r,device=dev,dtype=d)/r**.5
        def peft(): return x@w.T + (x@a.T)@b.T
        def fused_update(): return fused_lora(x,a,b,x@w.T)
        ref=peft(); got=fused_update(); err=(ref-got).abs().max().item()
        pt,ft=bench(peft),bench(fused_update)
        row={"m":m,"r":r,"base":base,"peft":pt,"base_plus_fused":ft,"max_abs":err,
             "layer_speedup":pt["median_us"]/ft["median_us"],
             "overhead_reduction_pct":100*(pt["median_us"]-ft["median_us"])/(pt["median_us"]-base["median_us"])}
        rows.append(row); print(json.dumps(row),flush=True)
    p=Path('artifacts/kernel_bench/full_linear.json');p.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
