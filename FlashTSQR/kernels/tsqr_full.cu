#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Batched TSQR, full operator (R + reflectors + Q@v). DEFAULT KERNEL.
//
// Merge nodes are TT-structured (triangle-on-triangle) AND packed:
//  * compute: column j touches only its j+2 active rows (~N^3/3 merge flops);
//  * storage: the two triangles live packed in shared memory (~N^2 floats
//    instead of the dense 2N^2), so at N=128 the merge tile drops from 131 KB
//    (1 block/SM) to ~67 KB (3 blocks/SM on GH200);
//  * inter-level R blocks and the merge reflectors travel packed through
//    global memory as well (~2x less traffic), and the Q@v application is
//    chunked over columns so large k no longer blows the shared-memory budget.
// See kernels/tsqr_dense_merge.cu for the dense-merge ablation baseline.

#define KC_CHUNK 64   // apply-phase column chunk

// ---------------- packed-triangle indexing -----------------------------------
__device__ __forceinline__ int tri(int k) { return (k * (k + 1)) >> 1; }
__device__ __forceinline__ int voff(int j) { return (j * (j + 3)) >> 1; }
__device__ __forceinline__ int tri_col(int t) {
    int k = (int)((sqrtf(8.f * (float)t + 1.f) - 1.f) * 0.5f);
    while (tri(k + 1) <= t) ++k;
    while (tri(k) > t) --k;
    return k;
}

// ---------------- reductions ------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ float block_sum(float v, float* smem) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    float r = smem[0];
    __syncthreads();
    return r;
}

// ---------------- dense Householder QR (leaves; input is dense) --------------
__device__ void hh_qr(float* tile, int rows, int N, float* vv, float* dots, float* red,
                      float* V, float* tau) {
    for (int j = 0; j < N; ++j) {
        float loc = 0.f;
        for (int i = j + threadIdx.x; i < rows; i += blockDim.x) {
            float x = tile[i * N + j];
            loc += x * x;
        }
        float nrm = sqrtf(block_sum(loc, red));
        float a0 = tile[j * N + j];
        float beta = (a0 >= 0.f) ? -nrm : nrm;
        if (nrm < 1e-30f) {
            if (V) { for (int i = threadIdx.x; i < rows; i += blockDim.x) V[i * N + j] = 0.f;
                     if (threadIdx.x == 0) tau[j] = 0.f; }
            __syncthreads(); continue;
        }
        for (int i = j + threadIdx.x; i < rows; i += blockDim.x)
            vv[i] = tile[i * N + j] - ((i == j) ? beta : 0.f);
        __syncthreads();

        float locv = 0.f;
        for (int i = j + threadIdx.x; i < rows; i += blockDim.x) locv += vv[i] * vv[i];
        float vnorm2 = block_sum(locv, red);
        if (vnorm2 < 1e-30f) {
            if (V) { for (int i = threadIdx.x; i < rows; i += blockDim.x) V[i * N + j] = 0.f;
                     if (threadIdx.x == 0) tau[j] = 0.f; }
            __syncthreads(); continue;
        }
        float t = 2.f / vnorm2;

        if (V) {
            for (int i = threadIdx.x; i < rows; i += blockDim.x)
                V[i * N + j] = (i >= j) ? vv[i] : 0.f;
            if (threadIdx.x == 0) tau[j] = t;
        }

        for (int k = j + threadIdx.x; k < N; k += blockDim.x) {
            float d = 0.f;
            for (int i = j; i < rows; ++i) d += vv[i] * tile[i * N + k];
            dots[k] = d;
        }
        __syncthreads();
        int cols = N - j, nr = rows - j;
        for (int idx = threadIdx.x; idx < nr * cols; idx += blockDim.x) {
            int i = j + idx / cols, k = j + idx % cols;
            tile[i * N + k] -= t * vv[i] * dots[k];
        }
        __syncthreads();
        if (threadIdx.x == 0) tile[j * N + j] = beta;
        for (int i = j + 1 + threadIdx.x; i < rows; i += blockDim.x) tile[i * N + j] = 0.f;
        __syncthreads();
    }
}

// ---------------- apply Q (leaves; dense reflectors), column-chunked ----------
__device__ void apply_Q(float* C, const float* V, const float* tau,
                        int rows, int N, int kc, float* dots) {
    for (int j = N - 1; j >= 0; --j) {
        float t = tau[j];
        if (t == 0.f) { __syncthreads(); continue; }
        for (int c = threadIdx.x; c < kc; c += blockDim.x) {
            float d = 0.f;
            for (int i = j; i < rows; ++i) d += V[i * N + j] * C[i * kc + c];
            dots[c] = d;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < (rows - j) * kc; idx += blockDim.x) {
            int i = j + idx / kc, c = idx % kc;
            C[i * kc + c] -= t * V[i * N + j] * dots[c];
        }
        __syncthreads();
    }
}

// ---------------- factor kernels ---------------------------------------------
__global__ void k_leaf(const float* __restrict__ B, float* Rp, float* V, float* tau,
                       int m, int N, int P, int rows) {
    extern __shared__ float sm[];
    float* tile = sm; float* vv = tile + rows * N; float* dots = vv + rows; float* red = dots + N;
    int mat = blockIdx.x / P, leaf = blockIdx.x % P;
    size_t off = ((size_t)mat * P + leaf);
    const float* src = B + (size_t)mat * m * N;
    for (int t = threadIdx.x; t < rows * N; t += blockDim.x) {
        int i = t / N;                       // zero-fill rows past m (padding)
        int gi = leaf * rows + i;
        tile[t] = (gi < m) ? src[(size_t)gi * N + (t % N)] : 0.f;
    }
    __syncthreads();
    hh_qr(tile, rows, N, vv, dots, red, V + off * (size_t)rows * N, tau + off * N);
    int T = tri(N);
    float* dst = Rp + off * (size_t)T;      // packed upper triangle out
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        int k = tri_col(t), i = t - tri(k);
        dst[t] = tile[i * N + k];
    }
}

// merge: packed triangles in, packed triangle out, packed reflectors out
__global__ void k_merge_tt(const float* __restrict__ Rin, float* Rout, float* V, float* tau,
                           int N, int pairs, int PinTotal, int PoutTotal) {
    extern __shared__ float sm[];
    int T = tri(N);
    float* P1 = sm; float* P2 = P1 + T; float* vv = P2 + T;
    float* dots = vv + (N + 1); float* red = dots + N;
    int mat = blockIdx.x / pairs, pair = blockIdx.x % pairs;
    size_t oi = ((size_t)mat * PinTotal + 2 * pair), oo = ((size_t)mat * PoutTotal + pair);
    const float* r0 = Rin + oi * (size_t)T;
    const float* r1 = Rin + (oi + 1) * (size_t)T;
    for (int t = threadIdx.x; t < T; t += blockDim.x) P1[t] = r0[t];
    for (int t = threadIdx.x; t < T; t += blockDim.x) P2[t] = r1[t];
    __syncthreads();

    size_t ov = ((size_t)mat * pairs + pair);       // V/tau exist only for merged pairs
    float* Vg = V + ov * (size_t)(2 * N) * N;       // dense [2N, N] reflectors (for WY apply)
    float* tg = tau + ov * N;
    for (int j = 0; j < N; ++j) {
        int nnz = j + 2;
        float loc = 0.f;
        for (int t = threadIdx.x; t < nnz; t += blockDim.x) {
            float x = (t == 0) ? P1[tri(j) + j] : P2[tri(j) + t - 1];
            loc += x * x;
        }
        float nrm = sqrtf(block_sum(loc, red));
        float a0 = P1[tri(j) + j];
        float beta = (a0 >= 0.f) ? -nrm : nrm;
        if (nrm < 1e-30f) {
            if (threadIdx.x == 0) tg[j] = 0.f;
            for (int t = threadIdx.x; t < 2 * N; t += blockDim.x) Vg[(size_t)t * N + j] = 0.f;
            __syncthreads(); continue;
        }
        for (int t = threadIdx.x; t < nnz; t += blockDim.x)
            vv[t] = (t == 0) ? (a0 - beta) : P2[tri(j) + t - 1];
        __syncthreads();
        float locv = 0.f;
        for (int t = threadIdx.x; t < nnz; t += blockDim.x) locv += vv[t] * vv[t];
        float vn2 = block_sum(locv, red);
        if (vn2 < 1e-30f) {
            if (threadIdx.x == 0) tg[j] = 0.f;
            for (int t = threadIdx.x; t < 2 * N; t += blockDim.x) Vg[(size_t)t * N + j] = 0.f;
            __syncthreads(); continue;
        }
        float tval = 2.f / vn2;
        for (int t = threadIdx.x; t < 2 * N; t += blockDim.x) {   // dense column j
            float val = 0.f;
            if (t == j) val = vv[0];
            else if (t >= N && t - N <= j) val = vv[t - N + 1];
            Vg[(size_t)t * N + j] = val;
        }
        if (threadIdx.x == 0) tg[j] = tval;

        for (int kc = j + 1 + threadIdx.x; kc < N; kc += blockDim.x) {
            const float* col2 = P2 + tri(kc);
            float d = vv[0] * P1[tri(kc) + j];
            for (int s = 1; s < nnz; ++s) d += vv[s] * col2[s - 1];
            dots[kc] = d;
        }
        __syncthreads();
        int cols = N - 1 - j;
        for (int t = threadIdx.x; t < nnz * cols; t += blockDim.x) {
            int s = t / cols, kc = j + 1 + t % cols;
            if (s == 0) P1[tri(kc) + j]     -= tval * vv[0] * dots[kc];
            else        P2[tri(kc) + s - 1] -= tval * vv[s] * dots[kc];
        }
        __syncthreads();
        if (threadIdx.x == 0) P1[tri(j) + j] = beta;
        __syncthreads();
    }
    float* dst = Rout + oo * (size_t)T;
    for (int t = threadIdx.x; t < T; t += blockDim.x) dst[t] = P1[t];
}

__global__ void k_unpack(const float* __restrict__ Rp, float* Rd, int N) {
    int mat = blockIdx.x;
    int T = tri(N);
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) {
        int i = t / N, k = t % N;
        Rd[(size_t)mat * N * N + t] = (i <= k) ? Rp[(size_t)mat * T + tri(k) + i] : 0.f;
    }
}

// ---------------- apply kernels (column-chunked) ------------------------------
__global__ void k_apply_merge_tt(const float* __restrict__ Wout, float* Win,
                                 const float* V, const float* tau,
                                 int N, int pairs, int PoutTotal, int PinTotal,
                                 int rank, int chunks) {
    extern __shared__ float sm[];
    int rows = 2 * N;
    int nodeIdx = blockIdx.x / chunks, ch = blockIdx.x % chunks;
    int mat = nodeIdx / pairs, node = nodeIdx % pairs;
    int c0 = ch * KC_CHUNK, kc = min(KC_CHUNK, rank - c0);
    float* C = sm; float* dots = C + (size_t)rows * KC_CHUNK;
    size_t o = ((size_t)mat * PoutTotal + node);
    size_t ov = ((size_t)mat * pairs + node);

    for (int t = threadIdx.x; t < N * kc; t += blockDim.x) {
        int i = t / kc, c = t % kc;
        C[i * kc + c] = Wout[o * (size_t)N * rank + (size_t)i * rank + (c0 + c)];
    }
    for (int t = N * kc + threadIdx.x; t < rows * kc; t += blockDim.x) C[t] = 0.f;
    __syncthreads();

    const float* Vg = V + ov * (size_t)(2 * N) * N;   // dense [2N, N]
    const float* tg = tau + ov * N;
    for (int j = N - 1; j >= 0; --j) {
        float t = tg[j];
        if (t == 0.f) { __syncthreads(); continue; }
        int nnz = j + 2;
        for (int c = threadIdx.x; c < kc; c += blockDim.x) {
            float d = Vg[(size_t)j * N + j] * C[j * kc + c];
            for (int s = 1; s < nnz; ++s)
                d += Vg[(size_t)(N + s - 1) * N + j] * C[(N + s - 1) * kc + c];
            dots[c] = d;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < nnz * kc; idx += blockDim.x) {
            int s = idx / kc, c = idx % kc;
            int i = (s == 0) ? j : (N + s - 1);
            float vij = (s == 0) ? Vg[(size_t)j * N + j] : Vg[(size_t)(N + s - 1) * N + j];
            C[i * kc + c] -= t * vij * dots[c];
        }
        __syncthreads();
    }
    size_t oi = ((size_t)mat * PinTotal + 2 * node);
    for (int t = threadIdx.x; t < N * kc; t += blockDim.x) {
        int i = t / kc, c = t % kc;
        Win[oi * (size_t)N * rank + (size_t)i * rank + (c0 + c)] = C[i * kc + c];
        Win[(oi + 1) * (size_t)N * rank + (size_t)i * rank + (c0 + c)] = C[(N + i) * kc + c];
    }
}

__global__ void k_apply_leaf(const float* __restrict__ W, float* Out,
                             const float* V, const float* tau,
                             int m, int N, int P, int rows, int rank, int chunks) {
    extern __shared__ float sm[];
    int nodeIdx = blockIdx.x / chunks, ch = blockIdx.x % chunks;
    int mat = nodeIdx / P, leaf = nodeIdx % P;
    int c0 = ch * KC_CHUNK, kc = min(KC_CHUNK, rank - c0);
    float* C = sm; float* dots = C + (size_t)rows * KC_CHUNK;
    size_t o = ((size_t)mat * P + leaf);

    for (int t = threadIdx.x; t < N * kc; t += blockDim.x) {
        int i = t / kc, c = t % kc;
        C[i * kc + c] = W[o * (size_t)N * rank + (size_t)i * rank + (c0 + c)];
    }
    for (int t = N * kc + threadIdx.x; t < rows * kc; t += blockDim.x) C[t] = 0.f;
    __syncthreads();
    apply_Q(C, V + o * (size_t)rows * N, tau + o * N, rows, N, kc, dots);
    float* dst = Out + (size_t)mat * m * rank;
    for (int t = threadIdx.x; t < rows * kc; t += blockDim.x) {
        int i = t / kc, c = t % kc;
        int gi = leaf * rows + i;
        if (gi < m) dst[(size_t)gi * rank + (c0 + c)] = C[i * kc + c];
    }
}

// ---------------- host API ---------------------------------------------------
#define WY_NB 32   // WY panel width for the blocked apply

struct Handle {
    torch::Tensor Rp, Vleaf, tleaf, Tleaf;
    std::vector<torch::Tensor> Vm, tm, Tm;
    std::vector<int> inCnt, outCnt;   // node counts per merge level (bye-aware)
    int m, N, P, rows;
};
static Handle g_h;

// T factors for panels of WY_NB reflectors: V [L, rows, N], tau [L, N]
// returns T [L, npan, nb, nb] with the last panel zero-padded.
static torch::Tensor build_T(torch::Tensor V, torch::Tensor tau, int N) {
    // Closed form: T^{-1} = diag(1/tau) + triu(Y, 1) with Y = V^T V, so each
    // panel is one batched bmm + one batched triangular solve (no column loop).
    int nb = WY_NB, npan = (N + nb - 1) / nb;
    auto L = V.size(0);
    auto Tt = torch::zeros({L, npan, nb, nb}, V.options());
    auto eye = at::eye(nb, V.options());
    for (int p = 0; p < npan; ++p) {
        int j0 = p * nb, w = std::min(nb, N - j0);
        auto Vp = V.narrow(2, j0, w);
        auto Y = at::bmm(Vp.transpose(1, 2), Vp);                  // [L, w, w]
        auto tp = tau.narrow(1, j0, w);                            // [L, w]
        auto invd = at::where(tp.abs() < 1e-30, at::full_like(tp, 1e30), tp.reciprocal());
        auto A = at::triu(Y, 1) + at::diag_embed(invd);            // [L, w, w] upper
        auto Tp = at::linalg_solve_triangular(A, eye.narrow(0, 0, w).narrow(1, 0, w)
                                                   .expand({L, w, w}),
                                              /*upper=*/true, /*left=*/true,
                                              /*unitriangular=*/false);
        Tt.select(1, p).narrow(1, 0, w).narrow(2, 0, w).copy_(Tp);
    }
    return Tt;
}

// blocked WY application of Q = prod_j H_j to C [L, rows, k], panels in reverse
static void wy_apply(torch::Tensor C, torch::Tensor V, torch::Tensor Tt, int N) {
    int nb = WY_NB, npan = (N + nb - 1) / nb;
    for (int p = npan - 1; p >= 0; --p) {
        int j0 = p * nb, w = std::min(nb, N - j0);
        auto Vp = V.narrow(2, j0, w);                        // [L, rows, w]
        auto W = at::bmm(Vp.transpose(1, 2), C);             // [L, w, k]
        W = at::bmm(Tt.select(1, p).narrow(1, 0, w).narrow(2, 0, w), W);
        C.sub_(at::bmm(Vp, W));
    }
}

torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb) {
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kFloat32 && B.dim() == 3);
    B = B.contiguous();
    int M = B.size(0), m = B.size(1), N = B.size(2);
    TORCH_CHECK(rows_per_leaf >= N, "rows_per_leaf must be >= N");
    int P = (int)((m + rows_per_leaf - 1) / rows_per_leaf);   // last leaf zero-padded
    auto o = B.options();
    int T = (N * (N + 1)) / 2, VT = (N * (N + 3)) / 2;

    auto Rp = torch::empty({M, P, T}, o);
    auto Vl = torch::empty({M, P, (int)rows_per_leaf, N}, o);
    auto tl = torch::empty({M, P, N}, o);
    size_t sh = (rows_per_leaf * N + rows_per_leaf + N + 32) * sizeof(float);
    cudaFuncSetAttribute(k_leaf, cudaFuncAttributeMaxDynamicSharedMemorySize, 227000);
    k_leaf<<<M * P, tpb, sh>>>(B.data_ptr<float>(), Rp.data_ptr<float>(),
                               Vl.data_ptr<float>(), tl.data_ptr<float>(), m, N, P, rows_per_leaf);

    g_h.Vm.clear(); g_h.tm.clear(); g_h.Tm.clear(); g_h.inCnt.clear(); g_h.outCnt.clear();
    size_t shm = (2 * T + (N + 1) + N + 32) * sizeof(float);
    cudaFuncSetAttribute(k_merge_tt, cudaFuncAttributeMaxDynamicSharedMemorySize, 227000);
    auto cur = Rp; int Pin = P;
    while (Pin > 1) {
        int pairs = Pin / 2, bye = Pin & 1, Pout = pairs + bye;
        auto nxt = torch::empty({M, Pout, T}, o);
        auto Vm = torch::empty({M, pairs, 2 * N, N}, o);
        auto tm = torch::empty({M, pairs, N}, o);
        k_merge_tt<<<M * pairs, tpb, shm>>>(cur.data_ptr<float>(), nxt.data_ptr<float>(),
                                            Vm.data_ptr<float>(), tm.data_ptr<float>(),
                                            N, pairs, Pin, Pout);
        if (bye)   // odd node passes through untouched
            nxt.select(1, Pout - 1).copy_(cur.select(1, Pin - 1));
        g_h.Vm.push_back(Vm); g_h.tm.push_back(tm);
        g_h.inCnt.push_back(Pin); g_h.outCnt.push_back(Pout);
        cur = nxt; Pin = Pout;
    }
    g_h.Rp = cur; g_h.Vleaf = Vl; g_h.tleaf = tl;
    g_h.Tleaf = torch::Tensor();          // T factors are built lazily on first apply
    g_h.m = m; g_h.N = N; g_h.P = P; g_h.rows = rows_per_leaf;

    auto Rd = torch::empty({M, N, N}, o);
    k_unpack<<<M, tpb>>>(cur.data_ptr<float>(), Rd.data_ptr<float>(), N);
    return Rd;
}

torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb) {
    v = v.contiguous();
    int M = v.size(0), N = g_h.N, rank = v.size(2), P = g_h.P, rows = g_h.rows, m = g_h.m;
    auto o = v.options();

    if (N < 160) {   // small N: the fused column-wise kernels win (low launch cost)
        int chunks = (rank + KC_CHUNK - 1) / KC_CHUNK;
        auto w = v.reshape({M, 1, N, rank}).contiguous();
        cudaFuncSetAttribute(k_apply_merge_tt, cudaFuncAttributeMaxDynamicSharedMemorySize, 227000);
        size_t shm = ((size_t)2 * N * KC_CHUNK + KC_CHUNK + 32) * sizeof(float);
        for (int L = (int)g_h.Vm.size() - 1; L >= 0; --L) {
            int pairs = g_h.Vm[L].size(1);
            int Pin = g_h.inCnt[L], Pout = g_h.outCnt[L];
            auto win = torch::empty({M, Pin, N, rank}, o);
            k_apply_merge_tt<<<M * pairs * chunks, tpb, shm>>>(
                w.data_ptr<float>(), win.data_ptr<float>(),
                g_h.Vm[L].data_ptr<float>(), g_h.tm[L].data_ptr<float>(),
                N, pairs, Pout, Pin, rank, chunks);
            if (Pin & 1)
                win.select(1, Pin - 1).copy_(w.select(1, Pout - 1));
            w = win;
        }
        auto out = torch::empty({M, m, rank}, o);
        cudaFuncSetAttribute(k_apply_leaf, cudaFuncAttributeMaxDynamicSharedMemorySize, 227000);
        size_t shl = ((size_t)rows * KC_CHUNK + KC_CHUNK + 32) * sizeof(float);
        k_apply_leaf<<<M * P * chunks, tpb, shl>>>(
            w.data_ptr<float>(), out.data_ptr<float>(),
            g_h.Vleaf.data_ptr<float>(), g_h.tleaf.data_ptr<float>(),
            m, N, P, rows, rank, chunks);
        return out;
    }

    // large N: blocked WY apply on cuBLAS batched GEMMs
    if (!g_h.Tleaf.defined()) {
        g_h.Tm.clear();
        for (size_t L = 0; L < g_h.Vm.size(); ++L) {
            int pairs = g_h.Vm[L].size(1);
            g_h.Tm.push_back(build_T(g_h.Vm[L].view({(int64_t)M * pairs, 2 * N, N}),
                                     g_h.tm[L].view({(int64_t)M * pairs, N}), N));
        }
        g_h.Tleaf = build_T(g_h.Vleaf.view({(int64_t)M * P, rows, N}),
                            g_h.tleaf.view({(int64_t)M * P, N}), N);
    }
    auto w = v.reshape({M, 1, N, rank}).contiguous();
    for (int L = (int)g_h.Vm.size() - 1; L >= 0; --L) {
        int pairs = g_h.Vm[L].size(1);
        int Pin = g_h.inCnt[L], Pout = g_h.outCnt[L];
        auto C = torch::zeros({(int64_t)M * pairs, 2 * N, rank}, o);
        C.narrow(1, 0, N).copy_(
            w.narrow(1, 0, pairs).reshape({(int64_t)M * pairs, N, rank}));
        wy_apply(C, g_h.Vm[L].view({(int64_t)M * pairs, 2 * N, N}), g_h.Tm[L], N);
        auto win = torch::empty({M, Pin, N, rank}, o);
        auto Cv = C.view({M, pairs, 2, N, rank});
        win.narrow(1, 0, 2 * pairs).view({M, pairs, 2, N, rank}).copy_(Cv);
        if (Pin & 1)
            win.select(1, Pin - 1).copy_(w.select(1, Pout - 1));
        w = win;
    }
    auto C = torch::zeros({(int64_t)M * P, rows, rank}, o);
    C.narrow(1, 0, N).copy_(w.reshape({(int64_t)M * P, N, rank}));
    wy_apply(C, g_h.Vleaf.view({(int64_t)M * P, rows, N}), g_h.Tleaf, N);
    return C.view({M, (int64_t)P * rows, rank}).narrow(1, 0, m).contiguous();
}
