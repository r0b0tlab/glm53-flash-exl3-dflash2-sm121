# Optimize profile — single GB10 (GLM-5.3-Flash EXL3, TP=1)

The maximal-performance configuration for `vllm_exl3_sm121` on one GB10
(SM121, 121.7 GiB unified), with measured numbers and the status of every
lever. AR plus the ported DFlash2 drafter (M4).

## Launch recipe (default profile)

```python
from vllm import LLM
llm = LLM(
    model='/home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3',
    quantization='exl3',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    max_num_seqs=16,                 # hybrid mamba cache with the drafter's KV group
    trust_remote_code=True,
    enforce_eager=False,             # CUDA graphs ON (target + speculator)
    compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16]},
    speculative_config={             # DFlash2 (M4). Drop for AR-only; then max_num_seqs=64.
        "method": "dflash",
        "model": "/home/r0b0tdgx/models/glm-5.3-flash-dflash2/exl3-3.00bpw",
        "num_speculative_tokens": 7,
    },
)
```

## Measured (same pack, same box, greedy)

| Workload | no spec | + DFlash2 K=7 | + n-gram spec |
|---|---|---|---|
| decode, repetitive filler (2k prompt, 256 new) | 17.96 tok/s | **50.16 tok/s (2.79x)** | 44.72 tok/s (2.49x) |
| decode, diverse prose (story, 256 new) | 18.98 tok/s | 17.29 tok/s (0.91x) | 11.04 tok/s (0.57x) |
| prefill (3608-token prompt, TTFT incl.) | 410.8 tok/s | ~unchanged | 410.4 tok/s |

DFlash2 acceptance telemetry (K=7): prose set **AL 2.65** (per-position
0.679/0.432/0.222/0.148/0.099/0.062/0.012, draft acceptance 23.6%); predictable
text **AL 6.60-7.42** (0.88-0.92 per position, 80-92%). Position-0 0.68+ =
capture/alignment correct. Lossless in class: temp-0 is true greedy
(seed matrix 6/6 identical, same-serve repeats byte-stable); spec-vs-no-spec
11 outputs = 6 byte-exact + 5 single-token near-tie flips (the documented
GB10 nondeterminism class, both continuations fluent). Receipt:
`docs/M4-RECEIPT.md`, logs `work/logs/m4-*`.

**DFlash2 verdict:** default ON — it dominates n-gram everywhere (2.79x vs
2.49x on repetitive, 0.91x vs 0.57x on prose) and helps code/repetitive text
most. The ~9% prose overhead is workload-driven (AL 2.65); K=4 is the untested
knob for prose-heavy serving.

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
| max_num_seqs=64 (AR) / 16 (DFlash2) | ON | hybrid mamba block budget incl. the drafter's KV group |
| n-gram spec decode | CONDITIONAL | superseded by DFlash2 where available; still a cheap option without a drafter |
| GEMV fast path (`EXL3_GEMV`) | DEFAULT | already active via the kernel's heuristic; forcing mode 2 at m=1 changed nothing on GB10 (memory-bound here — the kernel's own note) and shifted numerics, so the heuristic stays. Microbench: `work/logs/gemv_bench-20260919.py` |
| DFlash2 spec decode (M4) | **ON (default)** | EXL3 3.00bpw drafter (`glm-5.3-flash-dflash2/exl3-3.00bpw`); patches 0009-0012; measured above. `max_num_seqs=16` with the drafter |
| Recon tier (experts > 256 rows) | NOT PORTED | very long prefills fall back to the per-expert loop; banded path covers counts <= 256 |
| int8 GEMV (M2b) | NOT PORTED | dense-linears compute path; batch-1 decode is weight-traffic bound so the win is small |
| async scheduling | auto | SchedulerConfig default (None = on where supported) |
| TP=2 / EP=2 | PARKED | after the single-GB10 work, per operator decision |

## Explicit non-claims

No quality campaign has run on this engine (Q200v2 / NIAH / BFCL pending).
These are single-stream, greedy, 4k-context numbers on one GB10; they are not
a serving-under-load claim and not a comparison against the NVFP4 or 3090
lanes. The DFlash2 path has not been exercised at C>=2 (mixed prefill+decode).
