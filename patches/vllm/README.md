# vLLM patches

Rules (see AGENTS.md): anchored, fail-closed, minimal, every patch records its
upstream status. Applied by build tooling in order; a failed anchor stops the
build.

## Planned patch set (status: pending; anchors recorded at first cluster build)

| # | Working name | Why | Upstream status (2026-09-18) |
|---|---|---|---|
| 0001 | sm12x-sparse-mla-fallback | SM12x sparse-MLA instantiation gaps break long-context/vision paths (vLLM #56700 pitfall 2) | PR #54929 open, unmerged; we may carry a pure-torch fallback |
| 0002 | sm12x-graph-capture-guard | CUDA graph capture crash with certain ops >4K tokens on SM12x (#56700 pitfall 4) | none; guard/workaround ours |
| 0003 | deepgemm-warmup-skip | DeepGEMM autotune deadlock on first 128K+ batch (#56700 pitfall 1) | none; make skip/eager first-boot explicit |
| 0004 | nope-pad-guard | NoPE/padded-shard guards for aarch64/sm121 builds (pattern from our GLM package) | none |
| 0005 | engram-file-backed-hook | Allow Engram tables to be file-backed instead of host-RAM cpu_offload (UMA hosts) | #56357 closed unmerged; we carry it privately |
| 0006 | kda-fused-pack-layout | exllamav3-layout packs store KDA q\|k\|v fused (`qkv_proj`, `conv1d`); route them to the model's `in_proj_qkvbfg_a` / per-part conv params (qkv split owned by the EXL3 method); EXL3 packs keep KDA/MLA projections quantized (upstream forces BF16 for FP8 checkpoints) | none; pack-layout bridge for our converts |
| 0007 | sm120-nope-mla-pad | NoPE GLM-5.3 (`qk_rope_head_dim == 0`) on `FLASHINFER_MLA_SPARSE_SM120`: zero-pad the 64-wide rope lane on KV write and query (bit-exact NoPE; `concat_and_cache_mla` asserts pe_dim == 64) | derived from PR #53969 (hamiltongaianimd), validated on 2x GB10 in the GLM53-NVFP4 project |
| 0008 | sm90-nope-mla-sm121 | Offer `FLASHINFER_MLA_SPARSE_SM90` on capability 12 and run its FlashInfer wrapper with FA2 off Hopper: the SM120 backend cannot take a kpool-widened page table (no flashinfer dispatch for topk=2176) and its fp8_ds_mla layout is NoPE-hostile | port of the GLM53-NVFP4 P1 patch (proven serving GLM-5.3-Flash NVFP4 on 2x GB10); SM120+0007 stays as the fallback route |

0006, 0007 and 0008 are applied to `work/vllm` at the pinned base (`a00a3544b93e`) and their anchors are recorded in the patch files; the rest are pending until their anchors are recorded at first cluster build.
