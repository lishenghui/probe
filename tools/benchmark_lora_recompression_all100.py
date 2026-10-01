#!/usr/bin/env python3
"""Stream the 100-repository census through one recompression method."""
from __future__ import annotations

import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from benchmark_lora_recompression import (
    METHODS, Layer, batched_flash_fraq, batched_fraq, dense_svd, florist,
    locate_file, matching_up_key, matrix_pair, spectral, timed,
)
from analyze_lora_fraq_spectrum import load_flash_extension

ROOT = Path(__file__).resolve().parents[1]


def load_repo(adapter_root, order, repo_id, rows):
    output = []
    handles = {}
    try:
        for row in rows:
            path = locate_file(adapter_root, repo_id, row["filename"])
            if path is None:
                continue
            if path not in handles:
                handles[path] = safe_open(path, framework="pt", device="cpu")
                handles[path].__enter__()
            handle = handles[path]; keys = set(handle.keys())
            options = [f"{row['module']}{s}" for s in
                       (".lora_A.weight", ".lora_down.weight", ".lora.down.weight")]
            options += [k for k in keys if k.startswith(row["module"] + ".lora_A.") and k.endswith(".weight")]
            a_key = next((k for k in options if k in keys), None)
            match = matching_up_key(a_key) if a_key else None
            if not match or match[1] not in keys: continue
            pair = matrix_pair(handle.get_tensor(a_key), handle.get_tensor(match[1]))
            if pair is None: continue
            a, b = pair
            output.append(Layer(order, repo_id, row["filename"], row["module"], a, b, int(row["r95"])))
    finally:
        for h in handles.values(): h.__exit__(None, None, None)
    return output


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--adapter-root", type=Path, default=ROOT/"artifacts/hf_lora_census/top100_adapters")
    p.add_argument("--layers-csv", type=Path, default=ROOT/"artifacts/hf_lora_census/top100_rank32_64/layers.csv")
    p.add_argument("--rows-per-leaf", type=int, default=128)
    p.add_argument("--threads-per-block", type=int, default=256)
    p.add_argument("--max-flash-batch-size", type=int, default=128,
                   help="split FlashTSQR shape buckets to keep CUDA launch grids safe")
    p.add_argument("--debug-buckets", action="store_true")
    p.add_argument("--projects", type=int, default=100)
    p.add_argument("--repo-orders", type=int, nargs="+",
                   help="explicit popularity-order subset; overrides --projects")
    p.add_argument("--warmup", type=int, default=1); p.add_argument("--reps", type=int, default=3)
    args=p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    raw=list(csv.DictReader(args.layers_csv.open()))
    repo_rows=defaultdict(list); order={}
    for row in raw:
        if row["repo_id"] not in order: order[row["repo_id"]]=len(order)+1
        repo_rows[row["repo_id"]].append(row)
    if args.repo_orders:
        wanted_orders = set(args.repo_orders)
        if len(wanted_orders) != len(args.repo_orders) or min(wanted_orders) < 1 or max(wanted_orders) > len(order):
            raise ValueError("invalid or duplicate --repo-orders")
    else:
        if not 1 <= args.projects <= len(order): raise ValueError("invalid --projects")
        wanted_orders = set(range(1, args.projects + 1))
    selected_repos = {repo for repo, idx in order.items() if idx in wanted_orders}
    device=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=False
    ext=load_flash_extension(ROOT/"FlashTSQR/kernels/tsqr_full.cu") if args.method=="flashtsqr" else None
    layer_out=[]; repo_out=[]
    partial_path = args.output_dir / "repo_results.partial.csv"
    if partial_path.is_file():
        repo_out = list(csv.DictReader(partial_path.open()))
        completed = {int(x["repo_order"]) for x in repo_out}
        print(f"resuming from {partial_path}: {len(completed)} repos already complete", flush=True)
    else:
        completed = set()
    selected_layers = sum(len(repo_rows[x]) for x in selected_repos)
    print(f"GPU={torch.cuda.get_device_name(0)} method={args.method} repos={len(wanted_orders)} "
          f"repo_orders={sorted(wanted_orders)} census_layers={selected_layers}",flush=True)
    selected_items = [(repo, idx) for repo, idx in sorted(order.items(), key=lambda x:x[1]) if idx in wanted_orders]
    for subset_position, (repo_id, repo_order) in enumerate(selected_items, 1):
        if repo_order in completed:
            continue
        layers=load_repo(args.adapter_root,repo_order,repo_id,repo_rows[repo_id])
        repo_ms=0.0; batches=[]
        if args.method in ("fraq","flashtsqr"):
            buckets=defaultdict(list)
            for x in layers:
                o,i,r=x.shape; left=o<=i; buckets[(o if left else i,i if left else o,r,left)].append(x)
            raw_groups = list(buckets.values())
            groups = []
            for group in raw_groups:
                chunk = args.max_flash_batch_size if args.method == "flashtsqr" else len(group)
                groups.extend(group[start:start+chunk] for start in range(0, len(group), chunk))
            for group in groups:
                a=torch.stack([x.a for x in group]).to(device=device,dtype=torch.float32)
                b=torch.stack([x.b for x in group]).to(device=device,dtype=torch.float32)
                ks=[min(x.retained_rank,x.a.shape[0]) for x in group]
                if args.debug_buckets:
                    print(f"bucket repo={repo_order} batch={len(group)} A={tuple(a.shape)} "
                          f"B={tuple(b.shape)} kmax={max(ks)} left={b.shape[1] <= a.shape[2]}", flush=True)
                    torch.cuda.synchronize()
                # Census rank 0 denotes an all-zero LoRA update.  It is already
                # exactly compressed and must not launch applyQ with a zero-column
                # grid (CUDA rejects a launch with grid dimension zero).
                if max(ks) == 0:
                    ms = 0.0
                else:
                    fn=(lambda: batched_fraq(a,b,ks)) if args.method=="fraq" else \
                       (lambda: batched_flash_fraq(a,b,ks,ext,args.rows_per_leaf,args.threads_per_block))
                    ms,(_,_,_,_)=timed(fn,device,args.warmup,args.reps)
                repo_ms+=ms; batches.append(len(group))
                for x,k in zip(group,ks):
                    layer_out.append({"repo_order":repo_order,"repo_id":repo_id,"filename":x.filename,
                                      "module":x.module,"out_dim":x.shape[0],"in_dim":x.shape[1],
                                      "rank":x.shape[2],"retained_rank":k,"method":args.method,
                                      "batch_size":len(group),"batch_ms":ms,"amortized_ms":ms/len(group)})
                del a,b
        else:
            for x in layers:
                a=x.a.to(device=device,dtype=torch.float32); b=x.b.to(device=device,dtype=torch.float32)
                k=min(x.retained_rank,x.a.shape[0])
                fn=(lambda: dense_svd(a,b,k)) if args.method=="svd" else \
                   ((lambda: florist(a,b,k)) if args.method=="florist" else (lambda: spectral(a,b,k)))
                warmup=0 if args.method=="svd" else args.warmup
                reps=1 if args.method=="svd" else args.reps
                ms,_=timed(fn,device,warmup,reps); repo_ms+=ms; batches.append(1)
                layer_out.append({"repo_order":repo_order,"repo_id":repo_id,"filename":x.filename,
                                  "module":x.module,"out_dim":x.shape[0],"in_dim":x.shape[1],
                                  "rank":x.shape[2],"retained_rank":k,"method":args.method,
                                  "batch_size":1,"batch_ms":ms,"amortized_ms":ms})
                del a,b
        repo_out.append({"repo_order":repo_order,"repo_id":repo_id,"method":args.method,
                         "census_layers":len(repo_rows[repo_id]),"processed_layers":len(layers),
                         "buckets":len(batches),"max_batch_size":max(batches,default=0),
                         "latency_ms":repo_ms})
        write_csv(partial_path,repo_out)
        print(f"[{subset_position}/{len(selected_items)}; order={repo_order}] {repo_id}: "
              f"layers={len(layers)} buckets={len(batches)} "
              f"max_batch={max(batches,default=0)} latency={repo_ms:.3f} ms",flush=True)
        del layers; torch.cuda.empty_cache()
    write_csv(args.output_dir/"layer_results.csv",layer_out); write_csv(args.output_dir/"repo_results.csv",repo_out)
    lat=[float(x["latency_ms"]) for x in repo_out]
    summary={"method":args.method,"repos":len(repo_out),
             "layers":sum(int(x["processed_layers"]) for x in repo_out),
             "repo_latency_ms":{"mean":statistics.mean(lat),
                                "std":statistics.stdev(lat) if len(lat) > 1 else 0.0,
                                "median":statistics.median(lat),"min":min(lat),"max":max(lat)},
             "total_kernel_ms":sum(lat)}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
