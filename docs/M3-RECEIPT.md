# M3 receipt — first full EXL3 load + generation in this fork (vLLM / SM121, 2026-09-19)

**Status: ENGINE FUNCTIONAL.** The GLM-5.3-Flash EXL3 pack (`2.25hq-tapK3`,
91.67 GiB / 31 shards) loads through the `vllm_exl3_sm121` plugin on one GB10
and generates coherent, factually correct text. TP=1, `enforce_eager`, 4096
ctx. This is a bring-up receipt, not a performance or quality qualification.

## Runtime identity

- vLLM `a00a3544b93e` (v0.30.0rc1) + patches 0006–0008 (see `patches/vllm/`)
- exllamav3 v1.5.0 kernels vendored (`vendored/exllamav3/ext`), ext built
  `TORCH_CUDA_ARCH_LIST="12.0;12.1"`, CUDA 13.0
- flashinfer 0.6.18.post1, torch 2.13.0+cu130
- venv `~/venvs/exl3_sm121`; pack
  `/home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3`
- Logs: `work/logs/load-smoke-20260919.log`,
  `work/logs/quality-smoke-20260919.log`

## Blockers fixed this session (each one killed a prior load)

1. **MoE trellis slot geometry** (moe.py): `w13_trellis` was allocated
   `(288, 2*inter, in/16, out/16)` int16 = **72.0 GiB per layer** — the exact
   `cudaMalloc` that OOM'd the first load (`77309411328 bytes`). Slots now
   mirror the pack: `w13_trellis (e, 2, in/16, out/16, 16K)` with
   gate=`[e,0]`/up=`[e,1]`; `w2_trellis (e, in/16, out/16, 16K)`.
2. **Split-exact halves** (linear.py/moe.py): the pack's per-half `suh`
   input scales are not equal (LDLQ per-tensor), so a single fused trellis
   would be an approximation. Merged halves (`gate_up_proj`,
   `fused_qkv_a_proj`) and the MoE gate/up keep separate slots, one
   `exl3_gemm` each, concatenated on output.
3. **KDA fused pack layout** (patch 0006 + linear.py): the pack stores KDA
   `q|k|v` fused (`qkv_proj` EXL3, `conv1d` bf16; row order `[q|k|v]`
   verified byte-exact vs the BF16 source). `load_weights` routes them to
   the model's merged `in_proj_qkvbfg_a` (split into three EXL3 slots by the
   method) and to `q/k/v_conv1d`; EXL3 KDA/MLA projections stay quantized
   (upstream forces BF16 for FP8 checkpoints).
4. **lm_head prefix bridge** (linear.py): `language_model.lm_head` → pack key
   `lm_head`; the logits processor calls `quant_method.apply`, so the EXL3
   linear method serves the K=6 quantized head.
5. **NoPE sparse MLA** (patches 0007/0008): GLM-5.3 has
   `qk_rope_head_dim = 0`; the SM120 backend needs `pe_dim == 64` for
   `fp8_ds_mla` and has no flashinfer dispatch for the kpool-widened page
   table (topk=2176). Route taken: the SM90 sparse-MLA backend on
   capability 12 with FA2 (port of the GLM53-NVFP4 P1 route); 0007 (NoPE
   zero-pad on SM120) stays as the fallback.

## Evidence (excerpts from the logs)

```
Using FLASHINFER_MLA_SPARSE_SM90 attention backend out of potential backends:
  ['FLASHINFER_MLA_SPARSE_SM90', 'FLASHINFER_MLA_SPARSE_SM120']
Loading safetensors checkpoint shards: 100% Completed | 31/31 [10:54]
Available KV cache memory: 8.63 GiB
GPU KV cache size: 74,956 tokens (4096 ctx → 18.3x)
LOAD_OK
GEN_OK: ' 2+2 = 4. What is 3+3? ...'            (greedy, 512 tok, 50.9 s)
OUT: "The capital of France is Paris, with a population of approximately
      2.1 million people in the city proper and over 12 million in the
      metropolitan area."                      (greedy, 512 tok, 50.5 s)
PERF: 256 tokens in 33.39s = 7.67 tok/s        (~2k-token prompt, eager)
```

All three greedy sanity prompts are coherent; the arithmetic and the
Paris/population answer are correct. No U+FFFD, no NaNs, no kernel errors.

## Known gaps (explicit non-claims)

- **Performance**: MoE `apply()` is a per-expert Python loop (correct-first
  v1); `enforce_eager` (no CUDA graphs); TP=1. ~8–10 tok/s is a baseline,
  not a target. Fused/coop MoE kernels, int8 GEMV, graphs, and TP=2/EP are
  the next passes.
- **Quality**: no Q200v2 / BFCL / NIAH / vision campaigns have been run on
  this engine. The smoke is a coherence check only.
- **Context**: 4096 in this receipt; 262k (and 1M) unqualified.
- **Spec decode**: DFlash2 not wired (M4); MTP layers are skipped by design.
- **Vision**: tower loads (bf16, dense) but no image request has been run.
