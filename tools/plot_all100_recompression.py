#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(); p.add_argument("run_dir",type=Path); a=p.parse_args()
methods=("svd","florist","spectral","fraq","flashtsqr")
labels=("Dense SVD","FLoRIST","SpecTraL","FraQ","FlashTSQR")
colors=("#777777","#4c78a8","#f58518","#54a24b","#e45756")
data={m:list(csv.DictReader((a.run_dir/m/"repo_results.csv").open())) for m in methods}
stats={}
for m in methods:
    x=np.array([float(r["latency_ms"]) for r in data[m]])
    stats[m]={"mean_repo_latency_ms":float(x.mean()),"std_repo_latency_ms":float(x.std(ddof=1)),
              "median_repo_latency_ms":float(np.median(x)),"min_repo_latency_ms":float(x.min()),
              "max_repo_latency_ms":float(x.max())}

fig,axes=plt.subplots(1,2,figsize=(13,4.8)); pos=np.arange(5)
means=[stats[m]["mean_repo_latency_ms"] for m in methods]
stds=[stats[m]["std_repo_latency_ms"] for m in methods]
axes[0].bar(pos,means,yerr=stds,color=colors,capsize=4)
axes[0].set(yscale="log",ylabel="Latency per repository (ms)",title="Mean ± std across 100 HF LoRA repositories")
axes[0].set_xticks(pos,labels,rotation=20,ha="right"); axes[0].grid(axis="y",alpha=.25)
for j,(m,label,color) in enumerate(zip(methods,labels,colors)):
    values=np.array([float(r["latency_ms"]) for r in data[m]])
    jitter=np.linspace(-.08,.08,len(values)); axes[1].scatter(np.full(len(values),j)+jitter,values,s=9,alpha=.5,color=color)
    axes[1].boxplot([values],positions=[j],widths=.45,showfliers=False,patch_artist=True,
                    boxprops={"facecolor":color,"alpha":.25},medianprops={"color":"black"})
axes[1].set(yscale="log",ylabel="Latency per repository (ms)",title="Per-repository latency distribution")
axes[1].set_xticks(pos,labels,rotation=20,ha="right"); axes[1].grid(axis="y",alpha=.25)
fig.tight_layout(); fig.savefig(a.run_dir/"all100_repo_latency.png",dpi=240); plt.close(fig)
(a.run_dir/"all100_repo_latency_summary.json").write_text(json.dumps(stats,indent=2)+"\n")
print(json.dumps(stats,indent=2))
