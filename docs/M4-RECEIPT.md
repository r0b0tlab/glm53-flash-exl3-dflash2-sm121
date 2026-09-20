# M4 — DFlash2 speculative decoding on GLM-5.3-Flash EXL3 (single GB10)

Status: **working, lossless-in-class, telemetry-verified.** The drafter is the
`incoai/GLM-5.3-Flash-DFlash2` checkpoint, converted to an EXL3 3.00bpw pack
(`~/models/glm-5.3-flash-dflash2/exl3-3.00bpw`), loaded beside the target pack
(`exl3-2.25hq-tapK3`). All numbers below are TP=1, single stream, greedy,
4096 ctx, FULL CUDA graphs (target + speculator), on one GB10.

## What was built

1. **Target aux capture (patch 0009).** `Glm5NextModel` gains `EagleModelMixin`;
   the decoder loop captures tap states with the runner's id+1 semantics
   (`idx + 1 in aux_hidden_state_layers` = the OUTPUT of target layer idx),
   materializing the deferred mHC post state and contracting it
   (`hc_contract(hc_post(hidden, residual, post, comb), n)` — the stream mean
   the reference exports). `Glm5NextForCausalLM` / `ForConditionalGeneration`
   declare `SupportsEagle3`. Inert without a drafter.
   Verified: `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)`.
2. **Draft loader (patch 0010).** The DFlash2 draft loader resolves the draft's
   **own** quant config (a quantized drafter no longer inherits the target's
   pack).
3. **Draft pack support (plugin).** Lookup bridges for the draft's naming:
   bare-name resolution for `model.`-prefixed prefixes; the target-layer-offset
   shift (`model.layers.45..49` → pack `layers.0..4`, offset =
   `dflash_config.num_target_layers`); fused `qkv_proj` group (q/k/v slots);
   string shard ids `q`/`k`/`v` → slot index.
4. **Drafter KV group (patch 0012).** The GLM-5-Next KV fast path learns the
   drafter's `SlidingWindowSpec` layers: one extra group appended last
   (exact-fit block rescale when the geometry divides, else standalone compact
   tensors), threaded through bytes-per-block / config / max-memory. Without
   it the fast path bails to the generic unify and boot dies on MLA-vs-SWA page
   sizes. Inert without a drafter.
5. **Drafter context-KV weights (patch 0011).** `_build_context_kv_buffers`
   builds the fused KV projection from `qkv_proj.weight`; a quantized layer has
   none, so `Exl3LinearMethod.dense_weight()` materializes the operator's dense
   equivalent by running the wired `exl3_gemm` on an identity (fused
   reconstruct, scales/Hadamards folded). The plain `exl3_reconstruct` kernel
   emits the **had-transformed** basis and is unusable for this path.

Pack repairs (our artifact, recorded because they were load-blocking):
- the conversion had wrapped the two selector codebooks in `.weight` (renamed back);
- the data section had 5 gaps (safetensors rejects gaps) — rewritten contiguous;
- `dflash_config.num_target_layers: 45` added; `base_kernel` metadata recorded.

## Verified behavior

Acceptance telemetry (`disable_log_stats=False`, K=7):

| workload | mean acceptance length | per-position acceptance | draft acc |
|---|---|---|---|
| 8-prompt prose set | **2.65** | 0.679, 0.432, 0.222, 0.148, 0.099, 0.062, 0.012 | 23.6% |
| predictable text (cached repeats) | **6.60** | 0.883, 0.831, 0.818, 0.792, 0.766, 0.766, 0.740 | 80.0% |

Position-0 acceptance 0.68–0.88 → capture/alignment correct (the established
DFlash2 diagnostic: a position-0 deficit implicates capture, a healthy
position-0 with fast decay is workload).

Speed (same-session A/B):

| workload | no spec | DFlash2 K=7 | ratio |
|---|---|---|---|
| repetitive (2k→256) | 17.96 tok/s | **50.16 tok/s** | **2.79×** |
| diverse prose (256) | 18.98 tok/s | 17.29 tok/s | 0.91× |

Losslessness evidence:
- Same-seed cross-engine (spec vs no-spec, 11 outputs): 6 byte-exact (incl. the
  story), 5 single-token near-tie flips — identical prefixes of 94–300 chars,
  both continuations fluent. This is the documented GB10/SM121
  batch/kernel-shape nondeterminism class; cross-engine byte-parity is not a
  valid gate.
- Same-serve repeats are byte-stable (story_r2 == story_r3, math == math_r2).
- Seed matrix (one session, seeds 42/7/123 ×2, K=7): **6 runs, 1 unique output**
  — temp-0 is true greedy, seed-invariant in-session (`m4-final-verify-matrix-20260920.json`).
- Cross-session, same prompt: the matrix run's output is a byte-prefix of the
  validation run's (96 vs 128 token budgets); the no-spec twin's story equals
  the spec run's byte-for-byte. Same acceptance interval of the matrix run:
  AL **7.42**, per-position 0.917 ×7 (the drafter's ceiling on repeated text).

## Working config

```python
speculative_config={"method": "dflash", "model": DRAFT_PACK,
                    "num_speculative_tokens": 7}
max_num_seqs=16            # mamba cache budget with the drafter's KV group
cudagraph_capture_sizes=[1, 2, 4, 8, 16]
```

## Caveats / follow-ups

- **Prose overhead.** AL 2.65 on open prose makes K=7 ~neutral-to-slightly
  negative there (−9% here) while predictable/code-like text gains 2.8×.
  K=4 is the untested knob for prose-heavy serving.
- **C≥2.** The sibling overlay hit a selector-walk OOB class on mixed
  prefill+decode (NaN rows → sentinel id → target embed assert). Our tree has
  no walk kernel (torch `_score_edges`) but C≥2 has not been exercised here.
- Multi-stream/1M/NIAH were not part of this gate.

## Evidence files

- `work/logs/m4-dflash-spec-20260920.log` — first working spec serve (capture + sanity).
- `work/logs/m4-accept-telemetry-20260920.log` — SpecDecoding metrics.
- `work/logs/m4-validate-spec-20260920.log` / `m4-final-verify-spec-20260920.json` — 8-prompt set + self-repro control.
- `work/logs/m4-final-verify-nospec-20260920.json` — no-spec twin.
- `work/logs/m4-dflash-artwin-20260920.log` — first A/B twin.
- `work/logs/m4-final-verify-matrix-20260920.log` / `.json` — seed matrix (greedy proof).