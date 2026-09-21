# glm53-flash-exl3-dflash2-sm121

r0b0tlab's vLLM build for serving **EXL3 (ExLlamaV3 trellis) quants of
GLM-5.3-Flash** on **one NVIDIA GB10 / DGX Spark** (SM121, aarch64, CUDA 13,
ATS unified memory) — with DFlash2 speculative decoding. The full runtime:
pinned vLLM base + an out-of-tree EXL3 quantization plugin + vendored EXL3
CUDA kernels + anchored patches + measured receipts.

## Headline (single GB10, TP=1, temp 0, DFlash2 K=5; 2026-09-20)

| single-stream (median of 5, 2048 cap) | tok/s | | concurrency (structured, agg tok/s) | |
|---|---:|---|---:|---:|
| structured output | **50.2** | | ×1 | 46.8 |
| code | **43.4** | | ×4 | **66.5** |
| open prose | **19.5** | | ×16 | **67.1** |

Full tables, the K=5-vs-K=7 choice and measurement methods:
[`docs/RESULTS.md`](docs/RESULTS.md).

Losslessness and acceptance: see
[`docs/M4-RECEIPT.md`](docs/M4-RECEIPT.md) (greedy-exact in-session; spec-vs-AR
differences are single-token near-tie flips). The measured platform bound
(vLLM v1 host step; ~2 ms per D2H completion) and the two bugs fixed on the way
are in [`docs/CONCURRENCY-FINDING.md`](docs/CONCURRENCY-FINDING.md).

## What this is

- A pinned vLLM base (`runtime.lock.json`: vLLM v0.30.0rc1 @ a00a3544b93e)
  plus the `vllm_exl3_sm121` plugin registering `--quantization exl3`
  (dense linears, fused MoE, lm_head) and its CUDA extension.
- MIT-licensed EXL3 kernels vendored from turboderp-org/exllamav3 v1.5.0
  (see `vendored/VENDOR.md` for the exact subset), plus the fused MoE kernels
  (`exl3_moe` / `exl3_moe_gather`) and the EXL3 MoE GPU method.
- Anchored vLLM patches (`patches/vllm/`, fail-closed): SM90 sparse-MLA route
  on SM121, NoPE handling, KDA fused-pack layout, GLM DFlash2 aux capture,
  drafter quantization + drafter KV group — see `patches/vllm/README.md`.
- A GB10 placement planner (COPY / ATS-ALIAS / DISK decisions for unified
  memory budgets) and pack-metadata validation for the quantization contract.
- The exact serving recipe + lever-by-lever measurements:
  [`profiles/single-gb10-optimize.md`](profiles/single-gb10-optimize.md).

## Quickstart

```bash
# GB10 (aarch64, CUDA 13); build the extension + install the plugin
bash scripts/bootstrap_cluster.sh
vllm serve <glm-5.3-flash-exl3-pack> \
  --quantization exl3 --trust-remote-code \
  --served-model-name glm53-flash-exl3-dflash2 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 16 \
  --speculative-config '{"method":"dflash","model":"<dflash2-exl3-pack>","num_speculative_tokens":5}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,36,48,60,72,84,96]}' \
  --reasoning-parser glm47 --no-async-scheduling --max-num-batched-tokens 2048
```

Benchmarks: `work/bench/bench_class.py` (per-class medians + concurrency
ladder), `work/bench/kladder_curve.py` (curve + acceptance telemetry).

## Standard lanes (2026-09-21)

- Q200v2 text-180: **SCORED 170/180 = 94.4 %** (independent manual review on the
  manual-grade family; transport complete).
- BFCL v4 multi_turn_base structural-hard20: **10/20** (structured tool-call lane;
  decode-failure integration finding documented in `docs/RESULTS.md`).
- SM12X-LLM-BENCH systems: **complete / publishable** — NIAH 5/5 at the
  advertised 32768 window (incl. multi-key 33/66), concurrency and throughput
  rows, 832-sample telemetry. Full tables in `docs/RESULTS.md`.

## Status / limits

- Single GB10, TP=1. C≥2 tool-call loads not exercised; TP=2 parked.
- Vision: verified (image Q&A smoke on the published config — see
  `docs/RESULTS.md`; image requests are drafted text-only).
- The concurrency ceiling is the vLLM v1 host step on this platform (measured;
  see the finding doc) — kernels are not the limiter (tensor cores engaged,
  11–13% utilized, memory/issue-bound at these shapes).

## Governance

- `AGENTS.md` — non-negotiable rules (pins, provenance, license firewall, evidence).
- `runtime.lock.json` — single source of truth for versions.
- `docs/provenance.md` — where every vendored or adapted file comes from.
- `THIRD_PARTY_NOTICES.md` — third-party attribution.

## License

Our code: MIT (`LICENSE`). Vendored exllamav3 sources: MIT (`vendored/exllamav3/LICENSE`).
