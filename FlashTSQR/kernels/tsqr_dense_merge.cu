// Dense-merge batched TSQR -- ABLATION BASELINE ONLY.
// The default kernel is tsqr_full.cu (TT-structured merges, ~1.4-2.2x faster).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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

// ---------------- Householder QR of a [rows x N] shared tile -----------------
// Writes R into the upper triangle of the tile, and (if V/tau non-null) the
// reflectors: v_j in column j of V for rows [j,rows), tau_j = 2/||v_j||^2.
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
            if (V) { for (int i = j + threadIdx.x; i < rows; i += blockDim.x) V[i * N + j] = 0.f;
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
            if (V) { for (int i = j + threadIdx.x; i < rows; i += blockDim.x) V[i * N + j] = 0.f;
                     if (threadIdx.x == 0) tau[j] = 0.f; }
            __syncthreads(); continue;
        }
        float t = 2.f / vnorm2;

        if (V) {   // persist the reflector
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

// ---------------- apply Q = H_1...H_N  (reduced [rows x N]) to C [rows x rank]
__device__ void apply_Q(float* C, const float* V, const float* tau,
                        int rows, int N, int rank, float* dots, float* red) {
    for (int j = N - 1; j >= 0; --j) {        // reverse order applies Q (not Q^T)
        float t = tau[j];
        if (t == 0.f) { __syncthreads(); continue; }
        for (int k = threadIdx.x; k < rank; k += blockDim.x) {
            float d = 0.f;
            for (int i = j; i < rows; ++i) d += V[i * N + j] * C[i * rank + k];
            dots[k] = d;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < (rows - j) * rank; idx += blockDim.x) {
            int i = j + idx / rank, k = idx % rank;
            C[i * rank + k] -= t * V[i * N + j] * dots[k];
        }
        __syncthreads();
    }
}

// ---------------- factor kernels --------------------------------------------
__global__ void k_leaf(const float* __restrict__ B, float* R, float* V, float* tau,
                       int m, int N, int P, int rows) {
    extern __shared__ float sm[];
    float* tile = sm; float* vv = tile + rows * N; float* dots = vv + rows; float* red = dots + N;
    int mat = blockIdx.x / P, leaf = blockIdx.x % P;
    size_t off = ((size_t)mat * P + leaf);
    const float* src = B + (size_t)mat * m * N + (size_t)leaf * rows * N;
    for (int i = threadIdx.x; i < rows * N; i += blockDim.x) tile[i] = src[i];
    __syncthreads();
    hh_qr(tile, rows, N, vv, dots, red, V + off * rows * N, tau + off * N);
    float* dst = R + off * N * N;
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) {
        int i = t / N, k = t % N;
        dst[t] = (i <= k) ? tile[i * N + k] : 0.f;
    }
}

__global__ void k_merge(const float* __restrict__ Rin, float* Rout, float* V, float* tau,
                        int N, int Pin) {
    extern __shared__ float sm[];
    int rows = 2 * N;
    float* tile = sm; float* vv = tile + rows * N; float* dots = vv + rows; float* red = dots + N;
    int Pout = Pin / 2;
    int mat = blockIdx.x / Pout, pair = blockIdx.x % Pout;
    size_t oi = ((size_t)mat * Pin + 2 * pair), oo = ((size_t)mat * Pout + pair);
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) tile[t] = Rin[oi * N * N + t];
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) tile[N * N + t] = Rin[(oi + 1) * N * N + t];
    __syncthreads();
    hh_qr(tile, rows, N, vv, dots, red, V + oo * rows * N, tau + oo * N);
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) {
        int i = t / N, k = t % N;
        Rout[oo * N * N + t] = (i <= k) ? tile[i * N + k] : 0.f;
    }
}

// ---------------- apply kernels ---------------------------------------------
// tree level: w_out [M,Pout,N,rank] -> w_in [M,Pin=2*Pout,N,rank]
__global__ void k_apply_merge(const float* __restrict__ Wout, float* Win,
                              const float* V, const float* tau, int N, int Pout, int rank) {
    extern __shared__ float sm[];
    int rows = 2 * N;
    float* C = sm; float* dots = C + rows * rank; float* red = dots + rank;
    int mat = blockIdx.x / Pout, node = blockIdx.x % Pout;
    size_t o = ((size_t)mat * Pout + node);
    for (int t = threadIdx.x; t < N * rank; t += blockDim.x) C[t] = Wout[o * N * rank + t];
    for (int t = N * rank + threadIdx.x; t < rows * rank; t += blockDim.x) C[t] = 0.f;
    __syncthreads();
    apply_Q(C, V + o * rows * N, tau + o * N, rows, N, rank, dots, red);
    size_t oi = ((size_t)mat * (2 * Pout) + 2 * node);
    for (int t = threadIdx.x; t < N * rank; t += blockDim.x) Win[oi * N * rank + t] = C[t];
    for (int t = threadIdx.x; t < N * rank; t += blockDim.x) Win[(oi + 1) * N * rank + t] = C[N * rank + t];
}

// leaves: w [M,P,N,rank] -> out [M,m,rank]
__global__ void k_apply_leaf(const float* __restrict__ W, float* Out,
                             const float* V, const float* tau,
                             int m, int N, int P, int rows, int rank) {
    extern __shared__ float sm[];
    float* C = sm; float* dots = C + rows * rank; float* red = dots + rank;
    int mat = blockIdx.x / P, leaf = blockIdx.x % P;
    size_t o = ((size_t)mat * P + leaf);
    for (int t = threadIdx.x; t < N * rank; t += blockDim.x) C[t] = W[o * N * rank + t];
    for (int t = N * rank + threadIdx.x; t < rows * rank; t += blockDim.x) C[t] = 0.f;
    __syncthreads();
    apply_Q(C, V + o * rows * N, tau + o * N, rows, N, rank, dots, red);
    float* dst = Out + (size_t)mat * m * rank + (size_t)leaf * rows * rank;
    for (int t = threadIdx.x; t < rows * rank; t += blockDim.x) dst[t] = C[t];
}

// ---------------- host API ---------------------------------------------------
struct Handle {
    torch::Tensor R, Vleaf, tleaf;
    std::vector<torch::Tensor> Vm, tm;   // per merge level (leaf-side first)
    int m, N, P, rows;
};
static Handle g_h;

torch::Tensor tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb) {
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kFloat32 && B.dim() == 3);
    B = B.contiguous();
    int M = B.size(0), m = B.size(1), N = B.size(2);
    TORCH_CHECK(m % rows_per_leaf == 0);
    int P = m / rows_per_leaf;
    TORCH_CHECK((P & (P - 1)) == 0);
    auto o = B.options();

    auto R = torch::empty({M, P, N, N}, o);
    auto Vl = torch::empty({M, P, (int)rows_per_leaf, N}, o);
    auto tl = torch::empty({M, P, N}, o);
    size_t sh = (rows_per_leaf * N + rows_per_leaf + N + 32) * sizeof(float);
    cudaFuncSetAttribute(k_leaf, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000);
    k_leaf<<<M * P, tpb, sh>>>(B.data_ptr<float>(), R.data_ptr<float>(),
                               Vl.data_ptr<float>(), tl.data_ptr<float>(), m, N, P, rows_per_leaf);

    g_h.Vm.clear(); g_h.tm.clear();
    size_t shm = (2 * N * N + 2 * N + N + 32) * sizeof(float);
    cudaFuncSetAttribute(k_merge, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000);
    auto cur = R; int Pin = P;
    while (Pin > 1) {
        int Pout = Pin / 2;
        auto nxt = torch::empty({M, Pout, N, N}, o);
        auto Vm = torch::empty({M, Pout, 2 * N, N}, o);
        auto tm = torch::empty({M, Pout, N}, o);
        k_merge<<<M * Pout, tpb, shm>>>(cur.data_ptr<float>(), nxt.data_ptr<float>(),
                                        Vm.data_ptr<float>(), tm.data_ptr<float>(), N, Pin);
        g_h.Vm.push_back(Vm); g_h.tm.push_back(tm);
        cur = nxt; Pin = Pout;
    }
    g_h.R = cur; g_h.Vleaf = Vl; g_h.tleaf = tl;
    g_h.m = m; g_h.N = N; g_h.P = P; g_h.rows = rows_per_leaf;
    return cur.reshape({M, N, N});
}

// v: [M, N, rank] -> returns Q @ v : [M, m, rank]
torch::Tensor tsqr_applyQ(torch::Tensor v, int64_t tpb) {
    v = v.contiguous();
    int M = v.size(0), N = g_h.N, rank = v.size(2), P = g_h.P, rows = g_h.rows, m = g_h.m;
    auto o = v.options();

    auto w = v.reshape({M, 1, N, rank}).contiguous();   // at the root
    cudaFuncSetAttribute(k_apply_merge, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000);
    size_t shm = (2 * N * rank + rank + 32) * sizeof(float);
    for (int L = (int)g_h.Vm.size() - 1; L >= 0; --L) {   // root -> leaves
        int Pout = g_h.Vm[L].size(1);
        auto win = torch::empty({M, 2 * Pout, N, rank}, o);
        k_apply_merge<<<M * Pout, tpb, shm>>>(w.data_ptr<float>(), win.data_ptr<float>(),
                                              g_h.Vm[L].data_ptr<float>(),
                                              g_h.tm[L].data_ptr<float>(), N, Pout, rank);
        w = win;
    }
    auto out = torch::empty({M, m, rank}, o);
    cudaFuncSetAttribute(k_apply_leaf, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000);
    size_t shl = (rows * rank + rank + 32) * sizeof(float);
    k_apply_leaf<<<M * P, tpb, shl>>>(w.data_ptr<float>(), out.data_ptr<float>(),
                                      g_h.Vleaf.data_ptr<float>(), g_h.tleaf.data_ptr<float>(),
                                      m, N, P, rows, rank);
    return out;
}
