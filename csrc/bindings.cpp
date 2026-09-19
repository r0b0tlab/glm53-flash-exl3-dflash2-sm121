// vllm_exl3_sm121_ext — bindings for the vendored EXL3 kernels.
//
// Milestone M2: dense ops wired to the reference call pattern
// (BC_LinearEXL3::run_gr in r0b0tlab/exllamav3:
//  exl3_gemm(x, trellis, y, suh, xh_scratch, svh, K=-1, mcg, mul1, 0)).
// exl3_moe_gemm stays fail-closed until M3 defines the vLLM FusedMoE call
// convention (the upstream MoE path is the coop kernel with routing
// inputs, not a plain grouped GEMM).

#include <torch/extension.h>

#include <string>

#include "quant/exl3_gemm.cuh"
#include "quant/reconstruct.cuh"

namespace {

void check_dense_inputs(const char * op, const at::Tensor& x,
                        const at::Tensor& trellis) {
  TORCH_CHECK(x.is_cuda() && trellis.is_cuda(),
              "vllm_exl3_sm121_ext::", op, " requires CUDA tensors");
  TORCH_CHECK(x.is_contiguous(),
              "vllm_exl3_sm121_ext::", op,
              " requires contiguous input (strided views misread)");
  TORCH_CHECK(x.dim() >= 1, "vllm_exl3_sm121_ext::", op,
              " input must have at least one dim");
}

// Shared dense body: the reference forward uses exl3_gemm for m == 1 and
// m > 1 alike (no separate gemv call). K is always -1 (read K from trellis
// metadata), matching BC_LinearEXL3.
torch::Tensor dense_forward(const char* op, const at::Tensor& x,
                            const at::Tensor& trellis, const at::Tensor& suh,
                            const at::Tensor& svh, int64_t K, bool mcg,
                            bool mul1, int64_t n_out) {
  check_dense_inputs(op, x, trellis);
  TORCH_CHECK(n_out > 0, "vllm_exl3_sm121_ext::", op, " n_out must be > 0");
  (void)K;  // reference passes force_shape_idx=-1 unconditionally
  at::Tensor xs = x.reshape({-1, x.size(-1)});
  at::Tensor y =
      torch::empty({xs.size(0), n_out}, x.options().dtype(x.scalar_type()));
  at::Tensor xh = torch::empty_like(xs);
  exl3_gemm(xs, trellis, y, suh, xh, svh, -1, mcg, mul1, 0);
  std::vector<int64_t> shape = x.sizes().vec();
  shape.back() = n_out;
  return y.reshape(shape);
}

}  // namespace

torch::Tensor exl3_gemv(torch::Tensor x, torch::Tensor trellis,
                        torch::Tensor suh, torch::Tensor svh, int64_t K,
                        bool mcg, bool mul1, int64_t n_out) {
  return dense_forward("exl3_gemv", x, trellis, suh, svh, K, mcg, mul1, n_out);
}

torch::Tensor exl3_gemm(torch::Tensor x, torch::Tensor trellis,
                        torch::Tensor suh, torch::Tensor svh, int64_t K,
                        bool mcg, bool mul1, int64_t n_out) {
  return dense_forward("exl3_gemm", x, trellis, suh, svh, K, mcg, mul1, n_out);
}

torch::Tensor exl3_moe_gemm(torch::Tensor x, torch::Tensor trellis_ptrs,
                            torch::Tensor suh, torch::Tensor svh, int64_t K,
                            bool mcg, bool mul1, int64_t n_out) {
  (void)x;
  (void)trellis_ptrs;
  (void)suh;
  (void)svh;
  (void)K;
  (void)mcg;
  (void)mul1;
  (void)n_out;
  TORCH_CHECK(false,
              "vllm_exl3_sm121_ext::exl3_moe_gemm is not wired yet (M3). The "
              "upstream MoE path is the coop kernel with routing inputs; the "
              "vLLM FusedMoE call convention it must match is defined in M3.");
}

torch::Tensor exl3_reconstruct(torch::Tensor trellis, torch::Tensor suh,
                               torch::Tensor svh, int64_t K, bool mcg,
                               bool mul1, int64_t n_out, int64_t k_in) {
  TORCH_CHECK(trellis.is_cuda(), "vllm_exl3_sm121_ext::exl3_reconstruct "
                                 "requires a CUDA trellis tensor");
  TORCH_CHECK(n_out > 0 && k_in > 0, "vllm_exl3_sm121_ext::exl3_reconstruct "
                                     "needs positive dims");
  (void)suh;  // non-fused path folds scales inside the kernel
  (void)svh;
  at::Tensor w =
      torch::empty({k_in, n_out},
                   trellis.options().dtype(torch::kHalf));
  reconstruct(w, trellis, (int)K, mcg, mul1);
  return w;
}

std::string build_info() {
  return std::string("vllm_exl3_sm121_ext 0.2.0; M2 dense wired; moe fail-closed; ") +
         std::string("cuda ") + std::to_string(CUDA_VERSION);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using dense_fn_t = torch::Tensor (*)(torch::Tensor, torch::Tensor,
                                       torch::Tensor, torch::Tensor, int64_t,
                                       bool, bool, int64_t);
  m.def("exl3_gemv", static_cast<dense_fn_t>(&exl3_gemv),
        "EXL3 decode GEMV (m=1, via exl3_gemm)");
  m.def("exl3_gemm", static_cast<dense_fn_t>(&exl3_gemm),
        "EXL3 prefill/batch GEMM (m>1)");
  m.def("exl3_moe_gemm", &exl3_moe_gemm, "EXL3 grouped MoE GEMM (M3)");
  m.def("exl3_reconstruct", &exl3_reconstruct, "EXL3 dequantize for checks");
  m.def("build_info", &build_info, "build identification");
}
