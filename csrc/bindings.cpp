// vllm_exl3_sm121_ext — bindings for the vendored EXL3 kernels.
//
// Milestone status: skeleton. The op surface is fixed (mirrors
// src/vllm_exl3_sm121/kernels.py); each op currently fails closed with a
// pointer to the wiring work. The build itself is real: setup_ext.py compiles
// this file for sm_120/sm_121 and validates the aarch64/CUDA-13 toolchain.
//
// Wiring plan (milestone M2, on cluster):
//   1. add vendored translation units to setup_ext.py sources
//      (quant/exl3_gemv.cu, quant/exl3_gemm.cu, quant/exl3_moe*.cu, ...),
//   2. replace each TORCH_CHECK body with a call into the corresponding
//      exllamav3 entry point using the same argument order documented in
//      vendored/exllamav3/ext/quant/*.h,
//   3. gate with exl3_reconstruct correctness tests (tests to be added).

#include <torch/extension.h>

#include <string>

namespace {

torch::Tensor not_wired(const char * op, const char * hint) {
  TORCH_CHECK(false,
              "vllm_exl3_sm121_ext::", op, " is not wired yet (M2). ",
              "See csrc/bindings.cpp for the wiring plan; kernel source: ", hint);
}

}  // namespace

torch::Tensor exl3_gemv(torch::Tensor x, torch::Tensor trellis,
                        torch::Tensor suh, torch::Tensor svh, int64_t K,
                        bool mcg, bool mul1) {
  return not_wired("exl3_gemv", "vendored/exllamav3/ext/quant/exl3_gemv.cu");
}

torch::Tensor exl3_gemm(torch::Tensor x, torch::Tensor trellis,
                        torch::Tensor suh, torch::Tensor svh, int64_t K,
                        bool mcg, bool mul1) {
  return not_wired("exl3_gemm", "vendored/exllamav3/ext/quant/exl3_gemm.cu");
}

torch::Tensor exl3_moe_gemm(torch::Tensor x, torch::Tensor trellis_ptrs,
                            torch::Tensor suh, torch::Tensor svh, int64_t K,
                            bool mcg, bool mul1) {
  return not_wired("exl3_moe_gemm",
                   "vendored/exllamav3/ext/quant/exl3_moe.cu");
}

torch::Tensor exl3_reconstruct(torch::Tensor trellis, torch::Tensor suh,
                               torch::Tensor svh, int64_t K, bool mcg,
                               bool mul1, int64_t n_out, int64_t k_in) {
  return not_wired("exl3_reconstruct",
                   "vendored/exllamav3/ext/quant/reconstruct.cu");
}

std::string build_info() {
  return std::string("vllm_exl3_sm121_ext skeleton; arch list from build env");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("exl3_gemv", &exl3_gemv, "EXL3 decode GEMV (m=1)");
  m.def("exl3_gemm", &exl3_gemm, "EXL3 prefill/batch GEMM (m>1)");
  m.def("exl3_moe_gemm", &exl3_moe_gemm, "EXL3 grouped MoE GEMM");
  m.def("exl3_reconstruct", &exl3_reconstruct, "EXL3 dequantize for checks");
  m.def("build_info", &build_info, "build identification");
}
