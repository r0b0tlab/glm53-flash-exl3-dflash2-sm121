# vllm-exl3-sm121

r0b0tlab's vLLM build for serving **EXL3 (ExLlamaV3 trellis) quants of
GLM-5.3-Flash** on **one NVIDIA GB10 / DGX Spark** (SM121, aarch64, CUDA 13,
ATS unified memory) — with DFlash2 speculative decoding. The full runtime:
pinned vLLM base + an out-of-tree EXL3 quantization plugin + vendored EXL3
CUDA kernels + anchored patches + measured receipts.

## Headline (single GB10, TP=1, temp 0, DFlash2 K=7; 2026-09-20)

| single-stream (median of 5, 2048 cap) | | concurrency (structured, agg tok/s) | |
|---|---:|---|---:|
| structured output | **52.5 tok/s** | ×1 | 50.4 |
| code | **43.5 tok/s** | ×4 | **51.4** |
| open prose | **18.6 tok/s** | ×16 | **49.8** |

Context vs the published field (same model, EXL3, DFlash2 K=7):

| | this engine | MiaAI-Lab (2× GB10, 4bpw) | single-Spark recipe (2.05bpw) |
|---|---:|---:|---:|
| code single-stream | **43.5** | 35–44 | — |
| ladder ×4 | 51.4 | 67 | 182* |
| ladder ×8 | **50.8** | 8.5–29 | — |
| ladder ×16 | **49.8** | 6.8 | — |

\* active-stream convention. Full table + methods: [`docs/RESULTS.md`](docs/RESULTS.md).

The engine holds its single-stream rate across all 16 lanes; the published
2× GB10 lane collapses beyond ×4. Losslessness and acceptance: see
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
  --speculative-config '{"method":"dflash","model":"<dflash2-exl3-pack>","num_speculative_tokens":7}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,56,64,80,96,112,128]}' \
  --reasoning-parser glm47 --no-async-scheduling --max-num-batched-tokens 2048
```

Benchmarks: `work/bench/bench_class.py` (per-class medians + concurrency
ladder), `work/bench/kladder_curve.py` (curve + acceptance telemetry).

## Status / limits

- Single GB10, TP=1. Quality campaign (Q200v2 / BFCL hard-20) pending;
  C≥2 tool-call loads not exercised; TP=2 parked.
- The concurrency ceiling is the vLLM v1 host step on this platform (measured;
  see the finding doc) — kernels are not the limiter (tensor cores engaged,
  11–13% utilized, memory/issue-bound at these shapes).

## Governance

- `AGENTS.md` — non-negotiable rules (pins, provenance, license firewall, evidence).
- `runtime.lock.json` — single source of truth for versions.
- `docs/provenance.md` — where every vendored or adapted file comes from.
- `THIRD_PARTY_NOTICES.md` — third-party attribution.

## License

Our code: Apache-2.0 (`LICENSE`). Vendored exllamav3 sources: MIT (`vendored/exllamav3/LICENSE`).
