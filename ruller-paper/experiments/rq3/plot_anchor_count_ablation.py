#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path("artifacts/rq3/results")
files={0:"grid_alloc_lorare_disjoint_original.json",
       4:"grid_alloc_lorare_anchor_n4.json",
       8:"grid_alloc_lorare_anchor_n8.json",
       16:"grid_alloc_lorare_disjoint_anchor2.json"}
rows={}
for n,f in files.items():
    d=json.load(open(root/f))["budgets"]["3191"]["sct"]
    rows[n]={k:d[k] for k in ("mean","p10","worst")}
(root/"anchor_count_ablation.json").write_text(json.dumps(rows,indent=2)+"\n")

plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":7.5,
                     "axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(3.25,2.05))
x=list(rows)
for key,label,color,mark in [("worst","worst", "#d24b40","o"),
                             ("p10","$p10$", "#7b61a8","s"),
                             ("mean","mean", "#2878b5","^")]:
    ax.plot(x,[rows[n][key] for n in x],marker=mark,lw=1.5,ms=4,label=label,color=color)
ax.set_xticks(x);ax.set_xlabel("unlabeled calibration prompts per adapter")
ax.set_ylabel("retention $R$");ax.set_ylim(.2,.9);ax.grid(axis="y",alpha=.25)
ax.legend(frameon=False,ncol=3,loc="lower right")
fig.tight_layout(pad=.4)
out=Path("ruller-paper/figures/anchor_count_ablation.pdf")
fig.savefig(out,bbox_inches="tight");print('wrote',out)
