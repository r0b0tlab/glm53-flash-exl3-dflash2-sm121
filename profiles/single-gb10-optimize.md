# Optimize profile — single GB10 (GLM-5.3-Flash EXL3, TP=1)

The maximal-performance configuration for `vllm_exl3_sm121` on one GB10
(SM121, 121.7 GiB unified), with measured numbers and the status of every
lever. AR-only unless a row says otherwise (DFlash2 is not ported yet).

## Launch recipe (default profile)

```python
from vllm import LLM
llm = LLM(
    model='/home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3',
    quantization='exl3',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    max_num_seqs=64,                 # hybrid mamba cache exposes 125 blocks
    trust_remote_code=True,
    enforce_eager=False,             # CUDA graphs ON
    compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24]},
)
```

## Measured (same pack, same box, greedy)

| Workload | no spec | + n-gram spec | ratio |
|---|---|---|---|
| decode, repetitive filler (2k prompt, 256 new) | 17.95 tok/s | **44.72 tok/s** | 2.49x |
| decode, diverse prose (story, 256 new) | **19.21 tok/s** | 11.04 tok/s | 0.57x |
| prefill (3608-token prompt, TTFT incl.) | 410.8 tok/s | 410.4 tok/s | ~1.0 |

Progression of the default profile: per-expert loop 7.67 -> fused MoE 15.13 ->
+ graphs 16.80 -> + banded prefill + capture 1..24: 17.95 tok/s decode,
410 tok/s prefill.

**n-gram spec decode verdict:** workload-conditional. It is a 2.5x win on
repetitive or code-like output and a ~1.7x *loss* on diverse prose (draft
overhead, no acceptance). Not in the default profile. With spec enabled the
hybrid mamba budget shrinks (24 sequences max, vs 64 without).

Logs: `work/logs/profile-{single-gb10,ngram}-20260919.log`,
`work/logs/quality-graphs-20260919.log`.

## Levers, with status

| Lever | Status | Notes |
|---|---|---|
| Fused MoE kernel (`exl3_moe` + `exl3_moe_gather`, deterministic slots) | ON | 2.2x per-layer vs the loop; banded prefill launches (m_tile 16/32/64) |
| CUDA graphs FULL, capture sizes 1..24 | ON | capped at the fused reach (24 x top-8 = 216 <= 256 rows); requires capture-safe routing + post-load warmup |
| Post-load fused warmup | ON | creates the ext device context before capture (capture fails without it) |
| Capture-safe routing (`scatter_add_`, no `bincount`) | ON | `bincount` is not capturable |
| fp8 KV (auto), chunked prefill, prefix caching | ON | vLLM defaults for this model |
| max_num_seqs=64 (no spec) / 24 (spec) | ON | hybrid mamba block budget |
| n-gram spec decode | CONDITIONAL | enable per workload (repetitive/code); measured both directions above |
| DFlash2 spec decode (M4) | NOT PORTED | the next big lever: 2.25x measured on the sibling lane, and unlike n-gram it helps all workloads |
| Recon tier (experts > 256 rows) | NOT PORTED | very long prefills fall back to the per-expert loop; banded path covers counts <= 256 |
| int8 GEMV (M2b) | NOT PORTED | dense-linears compute path; batch-1 decode is weight-traffic bound so the win is small |
| async scheduling | auto | SchedulerConfig default (None = on where supported) |
| TP=2 / EP=2 | PARKED | after the single-GB10 work, per operator decision |

## Explicit non-claims

No quality campaign has run on this engine (Q200v2 / NIAH / BFCL pending).
These are single-stream, greedy, 4k-context numbers on one GB10; they are not
a serving-under-load claim and not a comparison against the NVFP4 or 3090
lanes.
