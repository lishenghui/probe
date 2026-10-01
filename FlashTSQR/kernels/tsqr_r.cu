
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---- block reduction over blockDim.x threads -------------------------------
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

// ---- in-place Householder QR of a [rows x N] tile held in shared memory -----
// Leaves R in the top NxN triangle. `tile` is row-major with leading dim N.
// NOTE: parameter order must match the shared-memory layout used by the kernels:
//       tile | vv | dots | red
__device__ void hh_qr(float* tile, int rows, int N, float* vv, float* dots, float* red) {
    for (int j = 0; j < N; ++j) {
        // ---- Householder vector for column j, rows [j, rows)
        float loc = 0.f;
        for (int i = j + threadIdx.x; i < rows; i += blockDim.x) {
            float x = tile[i * N + j];
            loc += x * x;
        }
        float nrm2 = block_sum(loc, red);
        float nrm = sqrtf(nrm2);
        float a0 = tile[j * N + j];
        float beta = (a0 >= 0.f) ? -nrm : nrm;
        if (nrm < 1e-30f) { __syncthreads(); continue; }

        // v = x - beta e1  (stored in vv[j..rows))
        for (int i = j + threadIdx.x; i < rows; i += blockDim.x)
            vv[i] = tile[i * N + j] - ((i == j) ? beta : 0.f);
        __syncthreads();

        float locv = 0.f;
        for (int i = j + threadIdx.x; i < rows; i += blockDim.x) locv += vv[i] * vv[i];
        float vnorm2 = block_sum(locv, red);
        if (vnorm2 < 1e-30f) { __syncthreads(); continue; }
        float scale = 2.f / vnorm2;

        // ---- w_k = v^T A[:,k]   for k in [j, N)
        for (int k = j + threadIdx.x; k < N; k += blockDim.x) {
            float d = 0.f;
            for (int i = j; i < rows; ++i) d += vv[i] * tile[i * N + k];
            dots[k] = d;
        }
        __syncthreads();

        // ---- rank-1 update  A -= scale * v w^T
        int cols = N - j, nr = rows - j;
        for (int t = threadIdx.x; t < nr * cols; t += blockDim.x) {
            int i = j + t / cols, k = j + t % cols;
            tile[i * N + k] -= scale * vv[i] * dots[k];
        }
        __syncthreads();

        // exact zeros below the diagonal + the known beta
        if (threadIdx.x == 0) tile[j * N + j] = beta;
        for (int i = j + 1 + threadIdx.x; i < rows; i += blockDim.x) tile[i * N + j] = 0.f;
        __syncthreads();
    }
}

// ---- leaf kernel: one block per (matrix, leaf) ------------------------------
__global__ void tsqr_leaf(const float* __restrict__ B, float* __restrict__ Rout,
                          int m, int N, int P, int rows) {
    extern __shared__ float sm[];
    float* tile = sm;                   // rows*N
    float* vv   = tile + rows * N;      // rows
    float* dots = vv + rows;            // N
    float* red  = dots + N;             // 32

    int mat = blockIdx.x / P, leaf = blockIdx.x % P;
    const float* src = B + (size_t)mat * m * N + (size_t)leaf * rows * N;
    for (int i = threadIdx.x; i < rows * N; i += blockDim.x) tile[i] = src[i];
    __syncthreads();

    hh_qr(tile, rows, N, vv, dots, red);

    // write the NxN R of this leaf
    float* dst = Rout + ((size_t)mat * P + leaf) * N * N;
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) {
        int i = t / N, k = t % N;
        dst[t] = (i <= k) ? tile[i * N + k] : 0.f;
    }
}

// ---- merge kernel: QR of [2N x N] formed by stacking two R's ----------------
__global__ void tsqr_merge(const float* __restrict__ Rin, float* __restrict__ Rout,
                           int N, int Pin) {
    extern __shared__ float sm[];
    int rows = 2 * N;
    float* tile = sm;
    float* vv   = tile + rows * N;
    float* dots = vv + rows;
    float* red  = dots + N;

    int Pout = Pin / 2;
    int mat = blockIdx.x / Pout, pair = blockIdx.x % Pout;
    const float* r0 = Rin + ((size_t)mat * Pin + 2 * pair) * N * N;
    const float* r1 = Rin + ((size_t)mat * Pin + 2 * pair + 1) * N * N;
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) tile[t] = r0[t];
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) tile[N * N + t] = r1[t];
    __syncthreads();

    hh_qr(tile, rows, N, vv, dots, red);

    float* dst = Rout + ((size_t)mat * Pout + pair) * N * N;
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) {
        int i = t / N, k = t % N;
        dst[t] = (i <= k) ? tile[i * N + k] : 0.f;
    }
}

// ---- host entry: returns R [M, N, N] ---------------------------------------
torch::Tensor tsqr_r(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb) {
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kFloat32 && B.dim() == 3);
    B = B.contiguous();
    int M = B.size(0), m = B.size(1), N = B.size(2);
    TORCH_CHECK(m % rows_per_leaf == 0, "m must be divisible by rows_per_leaf");
    int P = m / rows_per_leaf;
    TORCH_CHECK((P & (P - 1)) == 0, "leaf count must be a power of two");

    auto opts = B.options();
    auto Rbuf = torch::empty({M, P, N, N}, opts);

    size_t sh_leaf = (rows_per_leaf * N + rows_per_leaf + N + 32) * sizeof(float);
    cudaFuncSetAttribute(tsqr_leaf, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000);
    tsqr_leaf<<<M * P, tpb, sh_leaf>>>(B.data_ptr<float>(), Rbuf.data_ptr<float>(),
                                       m, N, P, rows_per_leaf);

    size_t sh_mrg = (2 * N * N + 2 * N + N + 32) * sizeof(float);
    cudaFuncSetAttribute(tsqr_merge, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000);
    auto cur = Rbuf;
    int Pin = P;
    while (Pin > 1) {
        auto nxt = torch::empty({M, Pin / 2, N, N}, opts);
        tsqr_merge<<<M * (Pin / 2), tpb, sh_mrg>>>(cur.data_ptr<float>(),
                                                   nxt.data_ptr<float>(), N, Pin);
        cur = nxt; Pin /= 2;
    }
    return cur.reshape({M, N, N});
}
