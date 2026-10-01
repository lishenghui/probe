import json,glob
import numpy as np
BUD={'land':[2289,1321,943],'cts':[19202,8811,4924],'lorare':[5880,3191,2683]}
F={'land':512,'cts':1536,'lorare':512}
SRC={  # (rung) -> (family, label template)
 'land':   {'Intra-Spec':('intraspec_eval','dense{b}'),'E2E-Spec':('spec512_eval','dense{b}'),
            'FuncDP':('funcdp0_eval','dense{b}'),'FRA':('fra_clean_eval','dense{b}')},
 'cts':    {'Intra-Spec':('intraspec_eval','dense{b}'),'E2E-Spec':('fra_rungs_eval','cts_e2e_spec_b{b}'),
            'FuncDP':('funcdp0_eval','dense{b}'),'FRA':('fra_clean_eval','dense{b}')},
 'lorare': {'Intra-Spec':('intraspec_eval','dense{b}'),'E2E-Spec':('fra_rungs_eval','lorare_e2e_spec_b{b}'),
            'FuncDP':('funcdp0_eval','dense{b}'),'FRA':('fra_clean_eval','dense{b}')}}
def dedup(fam,pool):
    d={}
    for f in sorted(glob.glob(f'{fam}_{pool}_*.json')):
        j=json.load(open(f))
        for r in (j if isinstance(j,list) else [j]):
            if isinstance(r,dict) and r.get('variants'): d[r.get('adapter') or r.get('task')]=r
    return d
def norm(a): return a.replace('_10templates','')
def ret(r,v): return v['retained'] if 'retained' in v else (v['metric']-r['metric_base'])/(r['metric_orig']-r['metric_base'])

OUT={}
for pool,buds in BUD.items():
    cand={b:{} for b in buds}                     # b -> adapter -> [(k,u,src)]
    method={b:{} for b in buds}                   # b -> rung -> {adapter:(k,u)}
    for rung,(fam,tmpl) in SRC[pool].items():
        D=dedup(fam,pool)
        for b in buds:
            method[b][rung]={}
            for a0,r in D.items():
                a=norm(a0)
                v=r['variants'][tmpl.format(b=b)]
                k=int(round(v['rank_frac']*F[pool])); u=ret(r,v)
                cand[b].setdefault(a,[]).append((k,u,rung)); method[b][rung][a]=(k,u)
    for b in buds:
        inp=json.load(open(f'candidate_union_inputs/{pool}_b{b}.json'))['adapters']
        method[b]['Uniform']={}
        for a,cs in inp.items():
            for c in cs:
                if c['source'] in ('Uniform','Threshold-grid Oracle'):
                    cand[b].setdefault(a,[]).append((c['k'],c['u'],c['source']))
                    if c['source']=='Uniform': method[b]['Uniform'][a]=(c['k'],c['u'])
    OUT[pool]=(cand,method,buds)
json.dump({p:{str(b):{a:v for a,v in OUT[p][0][b].items()} for b in OUT[p][2]} for p in OUT},
          open('/tmp/cand.json','w'))
for pool in OUT:
    cand,method,buds=OUT[pool]
    for b in buds:
        n=len(cand[b]); m=sorted({len(v) for v in cand[b].values()})
        print(f"{pool:7s} b={b:6d}  adapters={n:2d}  candidates/adapter={m}")

# ---------------- solve ----------------
def maxmean(A,B):
    NEG=-1e18; dp=np.full(B+1,NEG); dp[0]=0.0
    for a,cs in A.items():
        nd=np.full(B+1,NEG)
        for k,u,_ in cs:
            if k>B: continue
            src=dp[:B+1-k]+u
            np.maximum(nd[k:],src,out=nd[k:])
        dp=np.maximum.accumulate(nd)
    return dp[B]/len(A)

def maxworst(A,B):
    for t in sorted({u for cs in A.values() for _,u,_ in cs},reverse=True):
        tot=0; ok=True
        for cs in A.values():
            f=[k for k,u,_ in cs if u>=t-1e-12]
            if not f: ok=False;break
            tot+=min(f)
        if ok and tot<=B: return t
    return None

def maxp10(A,B,N):
    i=int(np.floor(0.1*(N-1)))          # p10 >= t is guaranteed if s[i] >= t
    best=None
    for t in sorted({u for cs in A.values() for _,u,_ in cs},reverse=True):
        need=[];cheap=[]
        for cs in A.values():
            f=[k for k,u,_ in cs if u>=t-1e-12]
            g=min(k for k,_,_ in cs)
            need.append(min(f) if f else None); cheap.append(g)
        sav=sorted([(need[j]-cheap[j]) for j in range(len(need)) if need[j] is not None],reverse=True)
        miss=sum(1 for x in need if x is None)
        if miss>i: continue
        tot=sum(x for x in need if x is not None)+sum(cheap[j] for j in range(len(need)) if need[j] is None)
        for s in sav[:i-miss]: tot-=s
        if tot<=B: best=t;break
    return best

print("\n==================== ORACLE ====================")
RES={}
for pool in OUT:
    cand,method,buds=OUT[pool]
    for b in buds:
        A=cand[b]; N=len(A)
        # budget self-check on every method
        for rung,alloc in method[b].items():
            sp=sum(k for k,_ in alloc.values())
            if sp>b: print(f"  !! {pool} b={b} {rung} spends {sp} > {b}")
        mm=maxmean(A,b); mw=maxworst(A,b); mp=maxp10(A,b,N)
        # p10 also from every stored allocation (all are in the union)
        for rung,alloc in method[b].items():
            us=np.array([u for _,u in alloc.values()])
            mp=max(mp,float(np.percentile(us,10)))
            mm=max(mm,float(us.mean())); mw=max(mw,float(us.min()))
        RES[(pool,b)]=(mm,mp,mw)
        print(f"{pool:7s} b={b:6d} N={N:2d}  ORACLE mean={mm:.3f} p10={mp:.3f} worst={mw:.3f}")
json.dump({f"{k[0]}|{k[1]}":v for k,v in RES.items()},open('/tmp/oracle_new.json','w'),indent=1)
