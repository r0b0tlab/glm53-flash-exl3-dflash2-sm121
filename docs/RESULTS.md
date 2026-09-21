# Results — GLM-5.3-Flash EXL3 + DFlash2 on a single GB10 (SM121)

All rows: one NVIDIA GB10 (DGX Spark), TP=1, temp 0, thinking at the
template's lightest setting (`reasoning_effort=low`), 32,768 ctx.
Measured 2026-09-20 with this runtime (vLLM v0.30.0rc1 base + patches 0001-0012;
see `runtime.lock.json`). Raw logs/JSONs under `work/logs/`.

## Headline — DFlash2 K=5 (the published config)

| single-stream (median of 5, 2048-token cap) | tok/s |
|---|---:|
| structured output (25-country JSON) | **50.2** |
| code (Python class) | **43.4** |
| open prose (200-word story) | **19.5** |

Concurrency ladder (structured prompt, aggregate tok/s over the round):

| lanes | 1 | 2 | 4 | 6 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| agg tok/s | 46.8 | 61.7 | **66.5** | 65.9 | 66.6 | **67.1** |

The ladder holds ~66–67 aggregate from ×2 through ×16.

## K choice (both measured, same serve flags)

| | structured | code | prose | C1 | C16 |
|---|---:|---:|---:|---:|---:|
| **K=5 (published)** | 50.2 | 43.4 | 19.5 | 46.8 | **67.1** |
| K=7 (trained block) | 52.5 | 43.5 | 18.6 | 50.4 | 49.8 |

K=7 is ~4 % ahead on structured single-stream; K=5 is ahead on prose, and the
K=5 ladder scales (C16 +35 % over K=7). Ship K=5; K=7 is one flag away
(`num_speculative_tokens`) for structured-only workloads.

## Speculative decoding

- Draft: `incoai/GLM-5.3-Flash-DFlash2` converted to an EXL3 3.00bpw pack
  (CC-BY-NC-ND source — not redistributed).
- Acceptance telemetry: prose AL 2.5–3.2 (position-0 acceptance 0.62–0.88);
  predictable/repetitive text AL 6.6–7.4 (draft acceptance 80–92 %).
- Greedy losslessness: in-session temp-0 is true greedy (seed matrix
  42/7/123 × 2 = 6/6 byte-identical; same-serve repeats byte-stable);
  spec-vs-no-spec on an 11-prompt set = 6 byte-exact + 5 single-token
  near-tie flips (the documented GB10 batch-shape nondeterminism class; both
  continuations fluent).

## Multimodal — vision smoke (live on the published config, 2026-09-20)

Two synthetic image probes through `/v1/chat/completions` returned exact
readings: text-in-image ("BANANA 42" in a red-bordered frame) and a
count/color/label question (3 circles — blue/green/orange left-to-right, plus
"HELLO WORLD"). The pack ships the 347-tensor vision tower (`model.visual.*`);
the engine wires the MM encoder (FLASH_ATTN) with a 32k-token encoder cache.
Image requests are drafted **text-only** — the DFlash2 drafter logs
"does not support external multimodal embeddings" and passes text draft inputs.

Reproduce: `work/bench/vision_probe.py`; evidence
`work/logs/vision-probe-20260920.log`. Vision was NOT part of the timing
ladder; the perf rows are text-only.

## Engine work behind these numbers

- Fused EXL3 MoE (`exl3_moe` / `exl3_moe_gather`, vendored exllamav3 v1.5.0
  kernels): 2.2× per-layer over the per-expert loop; banded prefill; capture-safe
  routing; post-load fused warmup; capture path above the fused row capacity
  rebuilt readback-free (capturing beyond ~32 tokens previously failed the boot).
- CUDA graphs FULL with lane-complete capture sizes (all (1+K)×lanes shapes to
  16 lanes).
- DFlash2 port: target aux capture with the mHC contraction; drafter KV group;
  drafter context-KV reconstruction; quantized-drafter pack support.
- Known bound (`docs/CONCURRENCY-FINDING.md`): the vLLM v1 host step on this
  platform (~2 ms per D2H completion behind in-flight work) sets the floor;
  kernels are not the limiter (tensor cores engaged, 11–13 % utilized,
  memory/issue-bound at these shapes).

## Reproduce

```bash
vllm serve <glm-5.3-flash-exl3-pack> \
  --quantization exl3 --trust_remote_code \
  --served-model-name glm53-flash-exl3-dflash2 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 16 \
  --speculative-config '{"method":"dflash","model":"<dflash2-exl3-pack>","num_speculative_tokens":5}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,36,48,60,72,84,96]}' \
  --reasoning-parser glm47 --no-async-scheduling --max-num-batched-tokens 2048
```

Bench: `work/bench/bench_class.py <tag> low`, `work/bench/kladder_curve.py`,
launch script `work/bench/serve-k5-q200.sh`.

## Evidence

- `work/logs/bench-class-k5.{log,json}` (published config) and
  `bench-class-k7.{log,json}` (K=7 variant).
- `work/logs/curve-*.log` — concurrency curves + flag A/Bs.
- `docs/M4-RECEIPT.md` — DFlash2 port, losslessness, acceptance.
- `docs/CONCURRENCY-FINDING.md` — host-step/D2H analysis; the two fixed bugs.
- `profiles/single-gb10-optimize.md` — lever list with per-lever receipts.
