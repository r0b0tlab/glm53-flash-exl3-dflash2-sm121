# vllm-exl3-sm121

r0b0tlab's own vLLM build for serving EXL3 (ExLlamaV3 trellis) quants on NVIDIA GB10 / DGX Spark (SM121, aarch64, CUDA 13, ATS unified memory).

Private development repository. Testing runs on the GB10 cluster, not on developer workstations.

## What this is

- A pinned vLLM base (see `runtime.lock.json`) plus an out-of-tree EXL3 quantization plugin (`vllm_exl3_sm121`) that registers `--quantization exl3`.
- MIT-licensed EXL3 kernels vendored from turboderp-org/exllamav3 v1.5.0 (see `vendored/VENDOR.md` for the exact subset and procedure).
- A GB10 placement planner (COPY / ATS-ALIAS / DISK decisions for unified memory budgets) and pack-metadata validation for our quantization contract.
- Build/bootstrap scripts that run on the cluster (`scripts/bootstrap_cluster.sh`).

## What this is not (yet)

- Not built on any machine as of this commit. The kernel bindings (`csrc/`), vLLM method classes (`linear.py`, `moe.py`, `embedding.py`) and the cluster build are the next work items; they require the GB10 cluster (aarch64, CUDA 13, sm_121).
- Not the DeepSeek-V4.1 campaign. Serving V4.1 is a later milestone; the base pin already carries the `vllm/models/deepseek_v41/` stack so no re-base is expected.

## Status matrix

| Component | State |
|---|---|
| Pack contract (`metadata.py`) | implemented, unit-tested (fixtures from our real packs) |
| Shard/TP/EP geometry (`geometry.py`) | implemented, unit-tested |
| GB10 placement planner (`placement.py`) | implemented, unit-tested |
| vLLM method classes (`config/linear/moe/embedding.py`) | interfaces written, on-cluster API validation pending |
| Kernel extension + bindings (`csrc/`) | skeleton, op list defined, build pending cluster |
| Cluster bootstrap (`scripts/`) | written, not yet executed |
| Container (`docker/`) | template, digest pinning pending cluster |

## Targets (our quants)

1. r0b0tlab Qwen3.8-27B EXL3 4.00bpw (dense) — first end-to-end target.
2. Its DFlash2 drafter EXL3 pack (external draft) — first spec-decode target.
3. r0b0tlab Qwen3.8-Flash-Next EXL3 2.50bpw (125B MoE + MTP + vision + 51B n-gram table) — hardest current pack.

## Quick start (on the GB10 cluster)

```bash
git clone https://github.com/r0b0tlab/vllm-exl3-sm121 && cd vllm-exl3-sm121
bash scripts/bootstrap_cluster.sh          # clones pinned vLLM, builds ext + plugin
# then, with a pack and its container:
#   see docs/BUILD-GB10.md and scripts/bootstrap_cluster.sh --help
```

## Governance

- `AGENTS.md` — non-negotiable rules (pins, provenance, license firewall, evidence).
- `runtime.lock.json` — single source of truth for versions.
- `docs/provenance.md` — where every vendored or adapted file comes from.
- `THIRD_PARTY_NOTICES.md` — third-party attribution.

## License

Our code: Apache-2.0 (`LICENSE`). Vendored exllamav3 sources: MIT (`vendored/exllamav3/LICENSE`).
