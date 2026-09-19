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
| 0006 | kda-fused-pack-layout | exllamav3-layout packs store KDA q\|k\|v fused (`qkv_proj`, `conv1d`); route them to the model's `in_proj_qkvbfg_a` / per-part conv params (qkv split owned by the EXL3 method) | none; pack-layout bridge for our converts |

0006 is applied to `work/vllm` at the pinned base (`a00a3544b93e`) and its anchors are recorded in the patch file; the rest are pending until their anchors are recorded at first cluster build.
