# Results — GLM-5.3-Flash EXL3 + DFlash2 on a single GB10 (SM121)

All rows: one NVIDIA GB10 (DGX Spark), TP=1, temp 0, thinking at the template's
lightest setting (`reasoning_effort=low`), DFlash2 K=7, 32768 ctx.
Measured 2026-09-20 with the M4 engine (see `runtime.lock.json` + patches
0001-0012). Raw logs and JSONs under `work/logs/`.

## Single-stream (median of 5, 2048-token cap)

| prompt class | this engine (1× GB10) | MiaAI-Lab (2× GB10, EXL3 4bpw) | single-Spark recipe (1× GB10, EXL3 2.05bpw) |
|---|---:|---:|---:|
| structured output | **52.5 tok/s** | — (code 35–44) | 64 |
| code | **43.5 tok/s** | 35–44 | — |
| open prose | **18.6 tok/s** | ~18 | 25 |

## Concurrency ladder (structured prompt, aggregate tok/s by concurrent streams)

| lanes | this engine (1× GB10) | MiaAI-Lab (2× GB10) | single-Spark recipe |
|---|---:|---:|---:|
| ×1 | 50.4 | 35 | ~64 |
| ×2 | 50.8 | 50 | — |
| ×4 | **51.4** | 67 | 182 (active-stream convention) |
| ×8 | **50.8** | 8.5–29 | — |
| ×16 | **49.8** | 6.8 | — |

This engine holds its single-stream rate across all 16 lanes; the published
2× GB10 EXL3 lane collapses beyond ×4. (Aggregate convention: total completion
tokens ÷ wall clock for the round, including scheduler gaps.)

## Speculative decoding (DFlash2 K=7)

- Draft: `incoai/GLM-5.3-Flash-DFlash2` converted to an EXL3 3.00bpw pack.
- Acceptance telemetry: prose AL 2.5–3.2 (position-0 acceptance 0.62–0.88,
  capture verified correct); predictable/repetitive text AL 6.6–7.4
  (draft acceptance 80–92%).
- Greedy losslessness: in-session temp-0 is true greedy (seed matrix 42/7/123 ×2
  = 6/6 byte-identical; same-serve repeats byte-stable); spec-vs-no-spec on an
  11-prompt set = 6 byte-exact + 5 single-token near-tie flips (the documented
  GB10 batch-shape nondeterminism class — both continuations fluent).

## Engine work behind these numbers

- Fused EXL3 MoE (`exl3_moe` + `exl3_moe_gather`, vendored exllamav3 v1.5.0
  kernels): 2.2× per-layer over the per-expert loop; banded prefill launches;
  capture-safe routing; post-load fused warmup.
- CUDA graphs FULL with lane-complete capture sizes (all (1+K)×lanes shapes to
  16 lanes); the quant MoE's capture path beyond the fused row capacity was
  rebuilt to stay readback-free (any capture > ~32 tokens previously failed the
  boot with `cudaErrorStreamCaptureUnsupported`).
- DFlash2 port: target aux capture with the mHC contraction; drafter KV group;
  drafter context-KV reconstruction; quantized-drafter pack support.
- Known bound (measured, documented in `docs/CONCURRENCY-FINDING.md`): the
  vLLM v1 host step (~100 ms at C1 on this platform; a D2H completion costs
  ~2 ms behind in-flight work) — which is why the ladder is flat, and why
  single-stream throughput = tokens-per-step ÷ host-step. In-class evidence:
  the same-served AR/spec runs and the sibling 2× GB10 numbers fit the same
  formula.

## Reproduce

```bash
# one GB10, aarch64, CUDA 13; venv with this repo + work/vllm pinned
vllm serve <glm-5.3-flash-exl3-pack> \
  --quantization exl3 --trust-remote-code \
  --served-model-name glm53-flash-exl3-dflash2 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 16 \
  --speculative-config '{"method":"dflash","model":"<dflash2-exl3-pack>","num_speculative_tokens":7}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,56,64,80,96,112,128]}' \
  --reasoning-parser glm47 --no-async-scheduling --max-num-batched-tokens 2048
```

Bench: `work/bench/bench_class.py <tag> low` (per-class medians + ladder),
`work/bench/kladder_curve.py` (curve + acceptance), `work/bench/serve-k5-q200.sh`
(the exact launch script).

## Evidence

- `work/logs/bench-class-k7.log` / `.json` — this table's rows (2026-09-20).
- `work/logs/curve-*.log` — concurrency curves (prose class) across flag A/Bs.
- `docs/M4-RECEIPT.md` — DFlash2 port + losslessness + acceptance.
- `docs/CONCURRENCY-FINDING.md` — the host-step/D2H analysis with nsys/py-spy
  evidence and the two fixed bugs.
- `profiles/single-gb10-optimize.md` — the lever list with per-lever receipts.
