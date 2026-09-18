# vendored/exllamav3 — vendored subset of turboderp-org/exllamav3

- Source: https://github.com/turboderp-org/exllamav3
- Pin: tag `v1.5.0`, commit `0740edc2da569fb99174023c1d2988b1e98cb41e`
- License: MIT (`LICENSE` in this directory, copied from the pinned tree)
- Vendored by: `scripts/vendor_upstream.sh` (authoritative; re-run to reproduce)

## Subset (ext/)

```
ext/quant/                      full EXL3 kernel tree (incl. hadamard):
  exl3_dq.cuh                     trellis decode
  exl3_gemv.* exl3_gemv_int8.*    decode GEMV paths (+int8 variant)
  exl3_gemm.* + inner/kernel      prefill GEMM
  exl3_moe.* (+coop variants)     grouped MoE GEMM/decode
  exl3_kernel_map.* codebook.cuh  codebook + dispatch
  pack.* quantize.* reconstruct.* quantizer side, used for validation tools
  hadamard.* util.*               rotations and utilities
  comp_units/                     per-K compilation units
ext/hgemm.* ext/hgemm_f16acc.cu
ext/graph.* ext/routing.*
ext/activation.* ext/activation_kernels.cuh
ext/compat.cuh ext/ptx.cuh ext/util.cuh ext/util.h
ext/reduction.cuh ext/histogram.*
```

Not vendored (present upstream, excluded on purpose): attention, cache, GDN,
sampling, stloader, ngram/ple, dsa_topk, dsv4_* (DeepSeek-V4-specific kernels —
v4.1 milestone), CPU/AVX targets, upstream bindings (`bindings.cpp` — we write
our own in `csrc/`), `cuda_drv`/`cuda_host` (revisit for ATS work).

## Rules

- Never edit vendored files in place. If a fix is required, put it in a patch
  or in `csrc/` and record it in `docs/provenance.md`.
- Re-running `scripts/vendor_upstream.sh` must produce a byte-identical tree
  (verify with `git status` after re-vendoring).
