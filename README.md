# glm53-flash-exl3-dflash2-sm121

r0b0tlab's vLLM build for serving **EXL3 (ExLlamaV3 trellis) quants of
GLM-5.3-Flash** on **one NVIDIA GB10 / DGX Spark** (SM121, aarch64, CUDA 13,
ATS unified memory) — with DFlash2 speculative decoding. The runtime is a
pinned vLLM source tree + an out-of-tree EXL3 quantization plugin + vendored
EXL3 CUDA kernels + anchored patches.

Weights: [`r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121`](https://huggingface.co/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121)
(98.5 GB / 91.8 GiB, 31 safetensors shards).

## Runtime status — read this before the numbers

**There is no published container image for this runtime, and the measurements
below were not produced inside one.** They came from a host-native install on a
single GB10: a vLLM source checkout at the pinned commit, the plugin installed
editable from this repo, and its CUDA extension built in place. The evidence is
in the logs — the serve tracebacks reference
`/home/r0b0tdgx/vllm-exl3-sm121/work/vllm/...` and
`/home/r0b0tdgx/vllm-exl3-sm121/src/vllm_exl3_sm121/moe.py`, and every run was
launched under a `systemd-run --user --scope` unit. No `docker://`, no `/opt`,
no site-packages frames appear in any serve log.

- **No GHCR package exists.** There is no `ghcr.io/r0b0tlab/vllm-exl3-sm121`
  and no container package for this repo. Do not write a `docker pull` for it.
- `docker/Dockerfile.stack` is an **unbuilt template**, not the measured
  runtime. Its base is still the literal placeholder
  `vllm/vllm-openai:PLACEHOLDER-PIN-ON-CLUSTER`, and no published vLLM
  `v0.30.0rc1` image exists on Docker Hub to point it at (the available
  `vllm/vllm-openai:v0.30.0` aarch64 image is built from commit `ced6857afa0e…`,
  not the pinned `a00a3544b93e`).
- The only container the measured evidence ever names is
  `sha256:9ab175696a13…` — that is `dsv41-tp4-sm121:overlay-v1-q200`, a
  **different project's** image, carried in the Q200 runner's `image-id` field.
  It is not this runtime and should not be cited as such.

So the reproducible path today is the source bootstrap below, not a container.

## Headline (single GB10, host-native, TP=1, temp 0, DFlash2 K=5; 2026-09-20)

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

- A pinned vLLM source tree (`runtime.lock.json`: vLLM v0.30.0rc1 @
  a00a3544b93e) plus the `vllm_exl3_sm121` plugin registering
  `--quantization exl3` (dense linears, fused MoE, lm_head) and its CUDA
  extension.
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

## Quickstart (host, GB10 / aarch64 / CUDA 13)

```bash
# Build vLLM @ the pinned commit, install the plugin, build the CUDA extension.
bash scripts/bootstrap_cluster.sh          # --check prints the pins and exits
```

That creates a venv (default `~/venvs/exl3_sm121`), checks out vLLM at
`a00a3544b93e` into `work/vllm`, installs it editable, installs this plugin
editable, builds the extension for `TORCH_CUDA_ARCH_LIST="12.0;12.1"`, and runs
the test suite plus a kernel probe.

Then serve, using the published pack and a local DFlash2 draft:

```bash
vllm serve /path/to/glm-5.3-flash-exl3-pack \
  --quantization exl3 --trust-remote-code \
  --served-model-name glm53-flash-exl3-dflash2 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 16 \
  --speculative-config '{"method":"dflash","model":"/path/to/dflash2-exl3","num_speculative_tokens":5}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,56,64,80,96,112,128]}' \
  --reasoning-parser glm47 --enable-auto-tool-choice --tool-call-parser glm47 \
  --no-async-scheduling --max-num-batched-tokens 2048
```

The `num_speculative_tokens: 5` and that `cudagraph_capture_sizes` list are the
measured K=5 config (from `work/bench/serve-k5-q200.sh` and the
`serve-k5-q200.log` banner). Benchmarks: `work/bench/bench_class.py` (per-class
medians + concurrency ladder), `work/bench/kladder_curve.py` (curve + acceptance
telemetry).

## Standard lanes (2026-09-21)

- Q200v2 text-180: **SCORED 170/180 = 94.4 %** (independent manual review on the
  manual-grade family; transport complete).
- BFCL v4 multi_turn_base structural-hard20: **10/20** (structured tool-call lane;
  decode-failure integration finding documented in `docs/RESULTS.md`).
- SM12X-LLM-BENCH systems: **complete / publishable** — NIAH 5/5 at the
  advertised 32768 window (incl. multi-key 33/66), concurrency and throughput
  rows, 832-sample telemetry. Full tables in `docs/RESULTS.md`.

## Status / limits

- **No container image is published.** See "Runtime status" above; a digest-pinned
  image would need a real `v0.30.0rc1` base built from the pinned commit, plus a
  model-load gate on a machine that still holds the pack.
- **Patch order:** apply `0007, 0008, 0009, 0010, 0011, 0012`. **`0006` must be
  skipped** — `0009` is a strict superset of it (it carries all three of 0006's
  hunks verbatim plus the DFlash2 aux-capture changes), so applying 0006 first
  makes 0009 fail to apply at `glm5next/nvidia/model.py`. Verified against the
  pinned commit: the six-patch order applies cleanly, all of 0006's effects are
  present in the result, and the patched file parses.
- `bootstrap_cluster.sh` checks out the pinned vLLM commit but does **not** apply
  `patches/vllm/` — apply them yourself before the first serve.
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

MIT — see [`LICENSE`](LICENSE).

- **Our code:** MIT. `pyproject.toml` declares it as the SPDX expression
  `license = "MIT"` (PEP 639) and ships both license files via `license-files`.
- **Vendored EXL3 kernels:** MIT, `vendored/exllamav3/LICENSE`
  (turboderp-org/exllamav3 v1.5.0 @ `0740edc2`, unmodified subset).
- **vLLM:** Apache-2.0, consumed as a runtime dependency and not vendored;
  `patches/vllm/` are diffs against it.
- Full attribution: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
  [`docs/provenance.md`](docs/provenance.md).
