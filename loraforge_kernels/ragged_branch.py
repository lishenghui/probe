"""Compact multi-LoRA branch with rank-zero early exit in both kernels."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _shrink(X,A,O,R,I,Z,K:tl.constexpr,RM:tl.constexpr,SK:tl.constexpr,BK:tl.constexpr,BR:tl.constexpr,SKIP:tl.constexpr):
 t=tl.program_id(0); part=tl.program_id(1); adapter=tl.load(I+t); r=tl.load(R+adapter)
 if not SKIP or r>0:
  off=tl.load(O+adapter); rr=tl.arange(0,BR); kk=part*tl.cdiv(K,SK)+tl.arange(0,BK)
  end=tl.minimum((part+1)*tl.cdiv(K,SK),K); acc=tl.full((BR,),0,tl.float32)
  for _ in range(tr.cdiv(tr.cdiv(K,SK),BK)):
   xv=tl.load(X+t*K+kk,kk<end,0)
   av=tl.load(A+(off+rr[:,None])*K+kk[None,:],(rr[:,None]<r)&(kk[None,:]<end),0)
   acc+=tl.sum(av*xv[None,:],1); kk+=BK
  tl.atomic_add(Z+t*RM+rr,acc,rr<r)

@tr.jit
def _expand(Y,Z,B,O,R,I,OUT,N:tl.constexpr,RM:tl.constexpr,BR:tl.constexpr,BN:tl.constexpr,SKIP:tl.constexpr):
 t=tl.program_id(0); nn=tl.program_id(1)*BN+tl.arange(0,BN)
 adapter=tl.load(I+t);r=tl.load(R+adapter);yv=tl.load(Y+t*N+nn,nn<N,0).to(tl.float32)
 if not SKIP or r>0:
  off=tl.load(O+adapter);rr=tl.arange(0,BR)
  z=tl.load(Z+t*RM+rr,rr<r,0)
  b=tl.load(B+(off+rr[None,:])*N+nn[:,None],(nn[:,None]<N)&(rr[None,:]<r),0)
  yv+=tl.sum(b*z[None,:].to(b.dtype),1)
 tl.store(OUT+t*N+nn,yv,nn<N)


def branch(x,y,a,b,offsets,ranks,indices,rmax,skip_zero=True,split_k=8):
 """A[sum ranks,K], B[sum ranks,N]; B includes adapter scale. No gathers."""
 if rmax==0:return y
 x=x.contiguous();y=y.contiguous();t,k=x.shape;n=y.shape[1]
 z=torch.zeros((t,rmax),device=x.device,dtype=torch.float32)
 # Sub-r16 widths are real, not silently rounded up to 16.
 br=max(1,tr.next_power_of_2(rmax))
 _shrink[(t,split_k)](x,a,offsets,ranks,indices,z,k,rmax,split_k,128,br,skip_zero,num_warps=4)
 out=torch.empty_like(y)
 _expand[(t,tr.cdiv(n,64))](y,z,b,offsets,ranks,indices,out,n,rmax,br,64,skip_zero,num_warps=4)
 return out
