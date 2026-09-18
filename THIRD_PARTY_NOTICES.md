# Third-party notices

This repository contains and derives from third-party work. Original notices remain in force for the material they cover.

## exllamav3 (vendored kernels)

- Project: turboderp-org/exllamav3
- License: MIT
- Source: https://github.com/turboderp-org/exllamav3
- Pinned: tag v1.5.0, commit 0740edc2da569fb99174023c1d2988b1e98cb41e
- Vendored subset: `vendored/exllamav3/ext/` (EXL3 quant/dequant/GEMV/GEMM/MoE kernels, Hadamard, graph, routing, activation, util). See `vendored/VENDOR.md` for the exact file list and reproduction procedure.
- License text: `vendored/exllamav3/LICENSE`

## vLLM

- Project: vllm-project/vllm
- License: Apache-2.0
- Pinned: tag v0.30.0rc1, commit a00a3544b93edbd66c8eda7285e4468f1202dc4b
- Consumed as a runtime dependency (not vendored). Patches live in `patches/vllm/`.

## Reference-only projects (no code copied)

The following AGPL-licensed or otherwise incompatible projects were used only as behavioral references. No code from them is included:

- vcruz305/vllm-exl3 (AGPL-3.0-only) — feature checklist and operational baseline only.
- dphnAI/sonar (AGPL-3.0) — design reference only.
- MiaAI-Lab and community DGX Spark recipe repositories (AGPL-3.0 current revisions) — measurement baselines only.
