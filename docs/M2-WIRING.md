# Milestone M2 — kernel wiring spec (from reference, 2026-09-19)

Reference: r0b0tlab/exllamav3 branch gb10 (community tip + aarch64 stubs),
SM121-native build, verified serving GLM-5.3-Flash EXL3 2.32bpw + DFlash2
(AL 6.39, 58.4 tok/s batch-1). All call patterns below are transcribed from
that tree (MIT, see docs/provenance.md).

## Dense forward (the only proven path)

`BC_LinearEXL3::run_gr` (`exllamav3/exllamav3_ext/libtorch/linear.cpp`):

- m == 1 (single row): `exl3_gemm_gr(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0, graph)`
- m > 1: `exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg, mul1, 0)` with
  `xh_ = empty_like(x)` scratch; graphs refused for m > 1.
- `y` is preallocated by the caller (`run_alloc`); K is passed as -1
  (read from trellis metadata).
- x MUST be contiguous (asserted in Python forward; strided views misread).

Consequence for our op surface: wire BOTH `exl3_gemv` and `exl3_gemm` to
`exl3_gemm` (m==1 included). The direct `exl3_gemv` entry
(`quant/exl3_gemv.cuh`) errors unless the call is "hard-eligible" — do not
use it for v1.

## int8 path (deferred to M2b, decision recorded)

- Dispatch is decided at MODEL LOAD, not per call (`exllamav3/model/config.py`):
  fused gate/up pairs are UNFUSED only when the separate calls can take the
  int8 path (mul1 codebook AND K below the per-arch cap;
  `ext.exl3_gemv_int8_max_k(device)`; Blackwell cap is K6).
- v1 wires the fused path only. M2b replicates the load-time fuse/unfuse
  decision in the method classes. Our GLM53 pack (K2-K5, mul1) is fully
  int8-eligible, so M2b matters for it.

## MoE (to mirror at implementation)

- Caller: `BC_*` in `exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp`
  calling into `quant/exl3_moe.cu`. Read that call site when wiring
  `exl3_moe_gemm`; whole-expert placement (no tensor splitting) per geometry.
- Contiguity + preallocated-y conventions match dense.

## Reconstruct (correctness gate)

- `exl3_reconstruct` -> `quant/reconstruct.cu`. Gate: reconstruct-MSE vs
  source on fixture tensors before any serve claim (receipts per AGENTS.md 5).

## Out of scope for M2

- CUDA graphs for m > 1 (refused upstream too).
- DFlash2 draft kernels in vLLM (separate milestone; external draft first).
