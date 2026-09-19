# M3c-perf receipt — fused MoE kernel wired (2026-09-19)

**Status: fused path landed and equivalence-checked.** The per-expert Python
loop is replaced by the vendored `exl3_moe` + `exl3_moe_gather` (the same
entry points the reference's `block_sparse_mlp.py` calls), in the
deterministic slot+gather mode (FUSED_DET).

## What changed

- `csrc/setup_ext.py`: the MoE translation units are compiled in (previously
  excluded until M3); ext is now `vllm_exl3_sm121_ext 0.3.0`.
- `csrc/bindings.cpp`: `exl3_moe`, `exl3_moe_gather`,
  `exl3_moe_max_concurrency` wired to the vendored kernels (the retired
  `exl3_moe_gemm` stub now points at the real path).
- `moe.py`: `apply()` dispatches to the fused path when
  `num_tokens * top_k <= rows` (256 with the mul1 wide tiles, else 128) and
  `top_k <= 32`; otherwise the per-expert loop (prefill-sized batches, until
  the tiled/banded tiers land). Pointer tables are built once from the
  stacked `(e, 2, ...)` storage; temp buffers `(C, R, hidden/inter)` fp16 with
  `C = exl3_moe_max_concurrency(0) = 6` on this GB10.

## Equivalence (real pack tensors, layer 3, experts 0..7)

`/tmp/moe_equiv_test.py` — same weights, same routings, loop vs fused:

```
tokens=1  topk=8: max_abs=0.000134 rel=0.009365
tokens=4  topk=8: max_abs=0.000023 rel=0.001542
tokens=16 topk=8: max_abs=0.000015 rel=0.001008
```

Agreement at fp16-rounding level (the loop accumulates fp16; the fused path
writes weighted fp32 rows and sums in k order).

## Timing (1 token, 8 experts per layer call, eager)

```
loop : 1.315 ms
fused: 0.597 ms      (2.2x)
```

At batch 1 the fused kernel is weight-traffic bound (~9.4 MB of trellis per
active expert); per-token MoE traffic is ~3.1 GB across 42 layers, which sets
the AR decode ceiling on this box.

## Open

- Prefill-sized batches still take the loop; the banded/tiled launches
  (count_lo/count_hi + m_tile 32/64) and the batched-reconstruct tier are the
  next step for prefill throughput.
- CUDA graphs: the fused `apply()` performs no host syncs and has static
  shapes, so PIECEWISE capture is expected to succeed now (the v1 loop's
  `unique/nonzero/tolist` invalidated the capture stream).
