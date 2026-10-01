#!/usr/bin/env python3
"""Fixed-budget fidelity and cross-proposal surrogate calibration."""
from __future__ import annotations
import argparse, csv, glob, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

COLORS={"LoRA Land":"#1b4f8f","Lots-of-LoRAs":"#2d7f5e","LoRARetriever":"#d1590a"}
INK,MUTED,GRID="#171717","#8d8c88","#e7e6e2"
plt.rcParams.update({"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb",
 "savefig.facecolor":"#fcfcfb","font.size":9.5,"axes.labelsize":10,
 "xtick.labelsize":9,"ytick.labelsize":9,"axes.edgecolor":MUTED,
 "axes.linewidth":.6,"legend.frameon":False,"figure.dpi":200})

CONFIG=[
 ("LoRA Land","fixedrho_land_*.json","functional_dp0_output_land12_*.json","rank0_spectral512_land12_*.json"),
 ("Lots-of-LoRAs","fixedrho_cts_*.json","functional_dp0_output_cts25_*.json","rank0_spectral_output_cts25_*.json"),
 ("LoRARetriever","fixedrho_lorare_*.json","functional_dp0_output_lorare_*.json","rank0_spectral_lorare_*.json")]

def load(root:Path,pat:str):
 out={}
 for p in glob.glob(str(root/pat)):
  d=json.loads(Path(p).read_text());out[d["adapter"]]=d
 return out

def surrogate(doc,ranks):
 tab={(int(x["module"]),int(x["k"])):float(x["d_js"]) for x in doc["single_layer"]}
 return sum(tab.get((i,int(k)),0.) for i,k in enumerate(ranks))

def collect(root:Path,tol=.02):
 fidelity=[]; matches=[]; adapters=[]
 for fleet,mp,fp,sp in CONFIG:
  meta,F,S=load(root,mp),load(root,fp),load(root,sp)
  for name in sorted(meta.keys()&F.keys()&S.keys()):
   d=meta[name]
   for budget in sorted({int(x["budget"]) for x in d["fixed_budget_allocations"]}):
    rows=[x for x in d["fixed_budget_allocations"] if int(x["budget"])==budget]
    rho=float(spearmanr([surrogate(d,x["module_ranks"]) for x in rows],
                        [float(x["d_js"]) for x in rows]).statistic)
    fidelity.append(dict(fleet=fleet,adapter=name,budget=budget,rho=rho))
   weights=np.asarray(d["module_costs"])
   spec=[]
   for x in S[name]["curve"]:
    ranks=tuple(x["module_ranks"])
    spec.append((int(weights@ranks),ranks,float(x["d_js"]),surrogate(d,ranks)))
   cells=[]
   for x in F[name]["curve"]:
    fr=tuple(x["module_ranks"]);fc=int(weights@fr)
    if not fc: continue
    sc,sr,sm,spred=min(spec,key=lambda z:(abs(z[0]-fc),z[0]))
    fpred,fm=surrogate(d,fr),float(x["d_js"])
    if abs(sc-fc)/fc>tol or sr==fr or min(spred,fpred,sm,fm)<=0: continue
    cells.append(dict(fleet=fleet,adapter=name,pred_ratio=spred/fpred,
                      measured_ratio=sm/fm,cost_ratio=sc/fc))
   matches.extend(cells)
   adapters.append(dict(fleet=fleet,adapter=name,
    pred_ratio=float(np.median([x["pred_ratio"] for x in cells])),
    measured_ratio=float(np.median([x["measured_ratio"] for x in cells])),
    cells=len(cells)))
 return fidelity,matches,adapters

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--results",type=Path,default=Path("artifacts/rq3/results"));ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
 fidelity,matches,adapters=collect(args.results);fleets=list(COLORS)
 fig,axes=plt.subplots(1,2,figsize=(9,3.55),constrained_layout=True);rng=np.random.default_rng(7)
 ax=axes[0]
 for i,fleet in enumerate(fleets):
  vals=np.asarray([x["rho"] for x in fidelity if x["fleet"]==fleet]); parts=ax.violinplot(vals,[i],widths=.72,showextrema=False)
  for b in parts["bodies"]:b.set_facecolor(COLORS[fleet]);b.set_edgecolor("none");b.set_alpha(.16)
  ax.scatter(i+rng.uniform(-.2,.2,len(vals)),vals,s=17,color=COLORS[fleet],alpha=.7,edgecolors="white",linewidths=.3)
  ax.plot([i-.22,i+.22],[np.mean(vals)]*2,color=COLORS[fleet],lw=2.3)
  ax.text(i,-.13,fr"mean $\rho={np.mean(vals):.3f}$",ha="center",va="top",fontsize=8,color=COLORS[fleet])
 ax.set_xticks(range(3),["LoRA\nLand","Lots-of-\nLoRAs","LoRA-\nRetriever"]);ax.set_ylim(-.18,1.03)
 ax.set_ylabel(r"within-budget ranking fidelity $\rho_i(K)$");ax.set_title("a   The surrogate ranks fixed-budget candidates",loc="left",weight="bold")
 ax=axes[1]
 for fleet in fleets:
  z=[x for x in adapters if x["fleet"]==fleet]
  ax.scatter([x["pred_ratio"] for x in z],[x["measured_ratio"] for x in z],s=30,color=COLORS[fleet],alpha=.8,edgecolors="white",linewidths=.4,label=fleet)
  n=sum(x["measured_ratio"]<1 for x in z)
  ax.text(.03,.95-.075*fleets.index(fleet),f"{fleet}: {n}/{len(z)} spectral-favoring",transform=ax.transAxes,color=COLORS[fleet],fontsize=8,va="top")
 lim=(.22,14);ax.plot(lim,lim,color=MUTED,ls=":",lw=1);ax.axhline(1,color=INK,ls="--",lw=1);ax.axvline(1,color=INK,ls="--",lw=1)
 ax.set_xscale("log");ax.set_yscale("log");ax.set_xlim(lim);ax.set_ylim(lim)
 ax.set_xlabel(r"surrogate ratio $\widetilde R^S/\widetilde R^F$")
 ax.set_ylabel(r"measured ratio $R^S/R^F$")
 ax.set_title("b   Calibration shifts across proposal shapes",loc="left",weight="bold");ax.legend(loc="lower right",fontsize=8)
 for ax in axes:
  ax.grid(color=GRID,lw=.7);ax.set_axisbelow(True)
  for s in ("top","right"):ax.spines[s].set_visible(False)
 args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,bbox_inches="tight");fig.savefig(args.output.with_suffix(".png"),bbox_inches="tight")
 with args.output.with_suffix(".csv").open("w",newline="") as h:
  w=csv.DictWriter(h,fieldnames=list(matches[0]),lineterminator="\n");w.writeheader();w.writerows(matches)
 print("fixed-budget means",{f:np.mean([x['rho'] for x in fidelity if x['fleet']==f]) for f in fleets})
 print("adapter spectral-favoring",{f:sum(x['measured_ratio']<1 for x in adapters if x['fleet']==f) for f in fleets})

if __name__=="__main__":main()
