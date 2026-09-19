# M2 dense receipt (2026-09-19, N1 GB10)

Extension: `vllm_exl3_sm121_ext 0.2.0` (`src/`, cuda 13000),
built `TORCH_CUDA_ARCH_LIST="12.0;12.1"`, venv `~/venvs/exl3_sm121`
(torch 2.13.0+cu130, vLLM 0.30.0rc1 a00a3544b93e).

## Procedure

Reference output (exllamav3 venv, torch 2.12.1):
`exllamav3/dflash2` gb10 @ 44636f0, `/tmp/ref_fwd.py` —
`LinearEXL3.forward` on `layers.0.mlp.down_proj`
(K=3, mcg=False, mul1=True from the live object),
x = `torch.randn(8, 12288, fp16, seed 0)` -> `/tmp/probe_y_ref.pt`.

Ours (this venv): `/tmp/ext-cmp.py` — `exl3_gemm` on the same pack
shards + same x.

## Result

- shapes (8, 4096) fp16 both sides
- max-abs-diff vs reference forward: 0.000000
- rel-diff vs reference forward: 0.000000

Pack: r0b0tlab GLM-5.3-Flash-EXL3-2.25hq-tapK3 (local
`/home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3`,
model-00001-of-00031).

## Notes

- A naive reconstruct-vs-bf16 comparison is NOT a valid check here
  (plain `reconstruct` emits Hadamard-basis weights by design); the
  kernel-vs-reference parity above is the correct receipt.
- `exl3_moe_gemm` remains fail-closed by design (M3).
