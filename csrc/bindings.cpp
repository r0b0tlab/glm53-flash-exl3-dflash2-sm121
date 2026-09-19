// vllm_exl3_sm121_ext — bindings for the vendored EXL3 kernels.
//
// M2: dense ops wired to the reference call pattern
// (BC_LinearEXL3::run_gr in r0b0tlab/exllamav3:
//  exl3_gemm(x, trellis, y, suh, xh_scratch, svh, K=-1, mcg, mul1, 0)).
// M3c-perf: the fused MoE path is wired directly to the vendored
// exl3_moe / exl3_moe_gather (the same entry points the reference's
// block_sparse_mlp.py calls), with the deterministic slot+gather
// accumulation (FUSED_DET) as the only supported mode here.

#include <torch/extension.h>

#include <string>

#include "quant/exl3_gemm.cuh"
#include "quant/exl3_moe.cuh"
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
              "vllm_exl3_sm121_ext::exl3_moe_gemm is retired; use exl3_moe + "
              "exl3_moe_gather (the fused kernel takes routing inputs).");
}

int64_t moe_max_concurrency(int64_t device) {
  return (int64_t) ::exl3_moe_max_concurrency((int) device);
}

// Fused MoE over experts with token counts in [count_lo, count_hi]:
// one cooperative kernel runs gate/up/down for every active expert group.
// output_scratch set => deterministic mode: each fused assignment writes a
// weighted row at fused_base[expert] + row; output_state is untouched.
void exl3_moe_op(torch::Tensor hidden_state, torch::Tensor output_state,
                 torch::Tensor expert_count, torch::Tensor token_sorted,
                 torch::Tensor weight_sorted, torch::Tensor temp_state_g,
                 torch::Tensor temp_state_u, torch::Tensor temp_intermediate_g,
                 torch::Tensor temp_intermediate_u, int64_t act_function,
                 int64_t K_gate, int64_t K_up, int64_t K_down,
                 torch::Tensor gate_ptrs_trellis, torch::Tensor gate_ptrs_suh,
                 torch::Tensor gate_ptrs_svh, torch::Tensor up_ptrs_trellis,
                 torch::Tensor up_ptrs_suh, torch::Tensor up_ptrs_svh,
                 torch::Tensor down_ptrs_trellis, torch::Tensor down_ptrs_suh,
                 torch::Tensor down_ptrs_svh, bool gate_mcg, bool gate_mul1,
                 bool up_mcg, bool up_mul1, bool down_mcg, bool down_mul1,
                 double act_limit, int64_t num_active,
                 c10::optional<torch::Tensor> output_scratch,
                 c10::optional<torch::Tensor> fused_base, int64_t count_lo,
                 int64_t count_hi, int64_t m_tile) {
  TORCH_CHECK(hidden_state.is_cuda() && hidden_state.is_contiguous(),
              "vllm_exl3_sm121_ext::exl3_moe requires a contiguous CUDA "
              "hidden_state");
  TORCH_CHECK(weight_sorted.scalar_type() == at::kHalf,
              "vllm_exl3_sm121_ext::exl3_moe requires fp16 weight_sorted");
  exl3_moe(hidden_state, output_state, expert_count, token_sorted,
           weight_sorted, temp_state_g, temp_state_u, temp_intermediate_g,
           temp_intermediate_u, (int) act_function, (int) K_gate, (int) K_up,
           (int) K_down, gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
           up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh, down_ptrs_trellis,
           down_ptrs_suh, down_ptrs_svh, gate_mcg, gate_mul1, up_mcg, up_mul1,
           down_mcg, down_mul1, (float) act_limit, (int) num_active,
           output_scratch, fused_base, (int) count_lo, (int) count_hi,
           (int) m_tile);
}

// Deterministic reduction: sum each token's top-k slots (in k order) from
// output_scratch into output_state (pre-zeroed fp32).
void exl3_moe_gather_op(torch::Tensor output_state, torch::Tensor output_scratch,
                        torch::Tensor flat_expert, torch::Tensor inv_order,
                        torch::Tensor expert_start, torch::Tensor slot_base,
                        torch::Tensor slot_kind, torch::Tensor weight_sorted) {
  TORCH_CHECK(output_state.is_cuda() && output_scratch.is_cuda(),
              "vllm_exl3_sm121_ext::exl3_moe_gather requires CUDA tensors");
  exl3_moe_gather(output_state, output_scratch, flat_expert, inv_order,
                  expert_start, slot_base, slot_kind, weight_sorted);
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
  return std::string("vllm_exl3_sm121_ext 0.3.0; M2 dense + M3c fused MoE; ") +
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
  m.def("exl3_moe_gemm", &exl3_moe_gemm,
        "retired stub; use exl3_moe + exl3_moe_gather");
  m.def("exl3_moe_max_concurrency", &moe_max_concurrency,
        "expert groups the fused MoE kernel can run concurrently");
  m.def("exl3_moe", &exl3_moe_op,
        "EXL3 fused MoE (gate/up/down over expert groups, deterministic "
        "scratch mode)");
  m.def("exl3_moe_gather", &exl3_moe_gather_op,
        "EXL3 MoE deterministic slot reduction into the output");
  m.def("exl3_reconstruct", &exl3_reconstruct, "EXL3 dequantize for checks");
  m.def("build_info", &build_info, "build identification");
}
