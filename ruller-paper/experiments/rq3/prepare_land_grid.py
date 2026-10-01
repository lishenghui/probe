#!/usr/bin/env python3
"""Assemble the seven-task LoRA Land grid and disjoint anchor files."""
import glob, json
from pathlib import Path
import numpy as np

root=Path("artifacts/rq3/results")
task=json.load(open(root/"predibase_margin.json"))+json.load(open(root/"predibase_cliff_glue.json"))
bases={r["adapter"]:r["metric_base"] for r in json.load(open(root/"predibase_warmup_binding.json"))}
for r in task:
    r["metric_base"]=bases[r["adapter"]]
    r["headroom"]=r["metric_orig"]-r["metric_base"]
(root/"land7_task_grid.json").write_text(json.dumps(task,indent=2)+"\n")

def token_mean(adapter,label,patterns):
    hits=[]
    for pattern in patterns:
        hits += glob.glob(str(root/pattern/f"{adapter}-{label}.npz"))
    if not hits: raise FileNotFoundError((adapter,label,patterns))
    return float(np.load(hits[0])["per_token_js"].mean())

full=[]
for r in task:
    variants={}
    dirs=["predibase_margin_tokens","predibase_cliff_glue_tokens"]
    for label,v in r["variants"].items():
        variants[label]={"L_W":v["L_W"],"rank_frac":v["rank_frac"],
                         "d_out":token_mean(r["adapter"],label,dirs)}
    full.append({"adapter":r["adapter"],"S":r["S"],"variants":variants})
(root/"land7_div_grid.json").write_text(json.dumps(full,indent=2)+"\n")

anchor=[]
for path in glob.glob(str(root/"land_anchor_disjoint_*.json")):
    for r in json.load(open(path)):
        variants={}
        d=Path(path).with_suffix("").with_name(Path(path).stem+"_tokens")
        for label,v in r["variants"].items():
            f=d/f"{r['adapter']}-{label}.npz"
            variants[label]={"L_W":v["L_W"],"rank_frac":v["rank_frac"],
                             "d_out":float(np.load(f)["per_token_js"].mean())}
        anchor.append({"adapter":r["adapter"],"S":r["S"],"variants":variants,
                       "prompts":r["n"],"example_start":r["example_start"]})
(root/"land7_anchor_disjoint.json").write_text(json.dumps(anchor,indent=2)+"\n")
print('task',len(task),'full',len(full),'anchor',len(anchor))
