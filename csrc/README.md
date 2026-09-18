# csrc

- `bindings.cpp` — op surface for `vllm_exl3_sm121_ext` (skeleton; ops fail
  closed until milestone M2 wires the vendored kernels).
- `setup_ext.py` — build entry point. Refuses to build unless
  `TORCH_CUDA_ARCH_LIST` targets sm_120/121 (GB10). Verifies the aarch64 +
  CUDA 13 toolchain on first cluster build.

Op surface (fixed; mirrored in `src/vllm_exl3_sm121/kernels.py`):
`exl3_gemv`, `exl3_gemm`, `exl3_moe_gemm`, `exl3_reconstruct`, `build_info`.

Do not edit `vendored/` here; patch or wrap instead (see AGENTS.md rule 9).
