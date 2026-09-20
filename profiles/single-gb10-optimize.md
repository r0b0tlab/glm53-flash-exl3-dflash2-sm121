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
        "num_speculative_tokens": 5,   # K sweep: 5 = balanced default; 7 = repetitive/code-heavy; 4 = prose-tight
    },
)
```

## Measured (same pack, same box, greedy)

| Workload | no spec | DFlash2 K=5 | DFlash2 K=7 | n-gram |
|---|---|---|---|---|
| decode, repetitive filler (2k prompt, 256 new) | 17.96 tok/s | 42.70 tok/s (2.38x) | **50.16 tok/s (2.79x)** | 44.72 tok/s (2.49x) |
| decode, diverse prose (story, 256 new) | 18.98 tok/s | **24.21 tok/s (1.28x)** | 17.29 tok/s (0.91x) | 11.04 tok/s (0.57x) |
| prefill (3608-token prompt, TTFT incl.) | 410.8 tok/s | ~unchanged | ~unchanged | 410.4 tok/s |

K sweep (same story + repetitive prompts, one run each; K=4: 22.50/40.22 =
1.19x/2.24x; prose-set average within noise of K=5):

| K | story | repetitive | acceptance (prose set) |
|---|---|---|---|
| 4 | 22.50 | 40.22 | AL 2.69, pos 0.70/0.45/0.31/0.24 |
| **5** | **24.21** | **42.70** | AL 2.45-3.18, pos-0 0.62-0.82 |
| 7 | 17.29 | 50.16 | AL 2.65, pos 0.68/0.43/0.22/0.15/0.10/0.06/0.01 |

The extra draft positions at K=7 accept at 10%/6%/1% on prose — pay verify+draft
cost for ~nothing; K=5 tightens the tail while keeping most of the repetitive
gain. Lossless in class: in-session temp-0 true greedy (K=7 seed matrix 6/6
identical; same-serve repeats byte-stable at K=7); spec-vs-no-spec 11 outputs =
6 byte-exact + 5 single-token near-tie flips (the documented GB10 nondeterminism
class, both continuations fluent — one K=4 near-tie landed in a repetition loop,
same class). Receipt: `docs/M4-RECEIPT.md`, logs `work/logs/m4-*`.
Concurrency smoke: C=4 mixed prefill+decode batch passes clean at K=4 and K=5
(sane outputs, no OOB); tool-heavy BFCL at C>=2 still unexercised.

**DFlash2 verdict:** default ON at **K=5** — beats n-gram everywhere and is a
win on both clean workloads (1.28x prose, 2.38x repetitive). K=7 for
repetitive/code-heavy serving (2.79x), K=4 as the prose-tight alternative.

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
| DFlash2 spec decode (M4) | **ON (default)** | EXL3 3.00bpw drafter (`glm-5.3-flash-dflash2/exl3-3.00bpw`); patches 0009-0012; K=5 balanced / K=7 repetitive / K=4 prose; `max_num_seqs=16` |
| Recon tier (experts > 256 rows) | NOT PORTED | very long prefills fall back to the per-expert loop; banded path covers counts <= 256 |
| int8 GEMV (M2b) | NOT PORTED | dense-linears compute path; batch-1 decode is weight-traffic bound so the win is small |
| async scheduling | auto | SchedulerConfig default (None = on where supported) |
| TP=2 / EP=2 | PARKED | after the single-GB10 work, per operator decision |

## Explicit non-claims

No quality campaign has run on this engine (Q200v2 / NIAH / BFCL pending).
These are single-stream, greedy, 4k-context numbers on one GB10; they are not
a serving-under-load claim and not a comparison against the NVFP4 or 3090
lanes. The DFlash2 path has not been exercised at C>=2 (mixed prefill+decode).
