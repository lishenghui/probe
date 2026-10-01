// Gate test: can a plain CUTLASS SM90 GEMM match cuBLAS on our layer shapes?
//
// The whole fused-sidecar plan rests on this.  Folding the LoRA expand into a
// GEMM epilogue only pays if the GEMM itself is cuBLAS-class -- the Triton
// attempt died here, running 1.4-1.9x slower before doing any sidecar work, so
// the epilogue traffic it saved cost more than it recovered.  No LoRA logic
// here on purpose: this measures the ceiling, nothing else.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using ElementIn = cutlass::bfloat16_t;
using ElementOut = cutlass::bfloat16_t;
using ElementAcc = float;

// x is [M, K] row-major; nn.Linear stores W as [N, K] row-major, which is the
// same bytes as a column-major [K, N] operand.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

static constexpr int kAlignIn = 128 / cutlass::sizeof_bits<ElementIn>::value;
static constexpr int kAlignOut = 128 / cutlass::sizeof_bits<ElementOut>::value;

template <class TileShape, class ClusterShape>
struct GemmFor {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc,
      ElementOut, LayoutC, kAlignOut,
      ElementOut, LayoutC, kAlignOut,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementIn, LayoutA, kAlignIn,
      ElementIn, LayoutB, kAlignIn,
      ElementAcc,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Op = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class G>
void run_gemm(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
  using Op = typename G::Op;
  int M = x.size(0), K = x.size(1), N = w.size(0);

  auto stride_a = cutlass::make_cute_packed_stride(
      typename Op::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      typename Op::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto stride_c = cutlass::make_cute_packed_stride(
      typename Op::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));

  typename Op::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const ElementIn*>(x.data_ptr()), stride_a,
       reinterpret_cast<const ElementIn*>(w.data_ptr()), stride_b},
      {{1.0f, 0.0f},
       reinterpret_cast<const ElementOut*>(out.data_ptr()), stride_c,
       reinterpret_cast<ElementOut*>(out.data_ptr()), stride_c}};

  Op op;
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t workspace_size = Op::get_workspace_size(args);
  auto workspace = torch::empty({static_cast<long>(workspace_size)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "cutlass cannot implement this shape");
  TORCH_CHECK(op.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess, "cutlass init failed");
  TORCH_CHECK(op.run(stream) == cutlass::Status::kSuccess, "cutlass run failed");
}

torch::Tensor cutlass_gemm(torch::Tensor x, torch::Tensor w, int64_t config) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "cuda tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "bf16 only");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "contiguous required");
  auto out = torch::empty({x.size(0), w.size(0)}, x.options());
  switch (config) {
    case 0: run_gemm<GemmFor<Shape<_128, _128, _64>, Shape<_1, _2, _1>>>(x, w, out); break;
    case 1: run_gemm<GemmFor<Shape<_128, _256, _64>, Shape<_1, _2, _1>>>(x, w, out); break;
    case 2: run_gemm<GemmFor<Shape<_256, _128, _64>, Shape<_2, _1, _1>>>(x, w, out); break;
    default: TORCH_CHECK(false, "unknown config");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_gemm", &cutlass_gemm, "plain CUTLASS SM90 GEMM (bf16)");
}
