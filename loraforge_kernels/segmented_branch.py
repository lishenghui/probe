"""Tensor-core LoRA for contiguous token segments, one segment per request.

Unlike token-wise SGMV this reuses factors across token tiles. Split-K partials
have disjoint writes: no atomic accumulation and no intermediate zero-fill.
The adapter mapping remains dynamic on device; packed factors stay resident.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _shrink(X,A,O,R,I,Z,T:tl.constexpr,K:tl.constexpr,RM:tl.constexpr,
            TOTAL:tl.constexpr,SK:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BR:tl.constexpr):
 request=tl.program_id(0);tile=tl.program_id(1);rp=tl.program_id(2)
 part=rp%SK;rt=rp//SK;adapter=tl.load(I+request);rank=tl.load(R+adapter)
 if rt*BR<rank:
  offset=tl.load(O+adapter)
  mm=tile*BM+tl.arange(0,BM);rr=rt*BR+tl.arange(0,BR)
  kk=part*tl.cdiv(K,SK)+tl.arange(0,BK)
  end=tl.minimum((part+1)*tl.cdiv(K,SK),K)
  acc=tl.full((BM,BR),0,tl.float32)
  for block in range(tl.cdiv(tl.cdiv(K,SK),BK)):
   x=tl.load(X+(request*T+mm[:,None])*K+kk[None,:],(mm[:,None]<T)&(kk[None,:]<end),0)
   a=tl.load(A+(offset+rr[None,:])*K+kk[:,None],(rr[None,:]<rank)&(kk[:,None]<end),0)
   acc+=tl.dot(x,a)
   kk+=BK
  tl.store(Z+(part*TOTAL+request*T+mm[:,None])*RM+rr[None,:],acc,(mm[:,None]<T)&(rr[None,:]<rank))


@tr.jit
def _expand(Y,Z,B,O,R,I,OUT,T:tl.constexpr,N:tl.constexpr,RM:tl.constexpr,
            TOTAL:tl.constexpr,SK:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BR:tl.constexpr):
 request=tl.program_id(0);mt=tl.program_id(1);nt=tl.program_id(2)
 adapter=tl.load(I+request);rank=tl.load(R+adapter);offset=tl.load(O+adapter)
 mm=mt*BM+tl.arange(0,BM);nn=nt*BN+tl.arange(0,BN);rr=tl.arange(0,BR)
 acc=tl.full((BM,BN),0,tl.float32)
 if rank>0:
  for tile in range(tl.cdiv(rank,BR)):
   z=tl.full((BM,BR),0,tl.float32)
   for part in tl.static_range(SK):
    z+=tl.load(Z+(part*TOTAL+request*T+mm[:,None])*RM+rr[None,:],(mm[:,None]<T)&(rr[None,:]<rank),0)
   b=tl.load(B+(offset+rr[:,None])*N+nn[None,:],(rr[:,None]<rank)&(nn[None,:]<N),0)
   acc+=tl.dot(z.to(b.dtype),b)
   rr+=BR
 y=tl.load(Y+(request*T+mm[:,None])*N+nn[None,:],(mm[:,None]<T)&(nn[None,:]<N),0)
 tl.store(OUT+(request*T+mm[:,None])*N+nn[None,:],y.to(tl.float32)+acc,(mm[:,None]<T)&(nn[None,:]<N))


def segmented_branch(x,y,a,b,offsets,ranks,request_ids,tokens_per_request,rmax,split_k=4):
 if rmax==0:return y
 x=x.contiguous();y=y.contiguous();total,k=x.shape;n=y.shape[1];requests=request_ids.numel()
 if total!=requests*tokens_per_request:raise ValueError('Invalid contiguous request segments')
 z=torch.empty((split_k,total,rmax),device=x.device,dtype=torch.float32)
 out=torch.empty_like(y)
 _shrink[(requests,tr.cdiv(tokens_per_request,32),tr.cdiv(rmax,16)*split_k)](
  x,a,offsets,ranks,request_ids,z,tokens_per_request,k,rmax,total,split_k,32,64,16,num_warps=4)
 _expand[(requests,tr.cdiv(tokens_per_request,32),tr.cdiv(n,64))](
  y,z,b,offsets,ranks,request_ids,out,tokens_per_request,n,rmax,total,split_k,32,64,16,num_warps=4)
 return out
