# Optimize profile — single GB10 (GLM-5.3-Flash EXL3, TP=1)

The maximal-performance configuration for `vllm_exl3_sm121` on one GB10
(SM121, 121.7 GiB unified). Every lever below is either ON in this profile,
measured-and-parked, or explicitly not ported yet.

## Launch recipe

```bash
source ~/venvs/exl3_sm121/bin/activate
python - <<'PY'
from vllm import LLM
llm = LLM(
    model='/home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3',
    quantization='exl3',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    max_num_seqs=64,                # hybrid mamba cache exposes 125 blocks
    trust_remote_code=True,
    enforce_eager=False,            # CUDA graphs ON
    compilation_config={
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24],
    },
)
PY
```

## Levers, with status

| Lever | Status | Notes |
|---|---|---|
| Fused MoE kernel (`exl3_moe` + `exl3_moe_gather`, deterministic slots) | ON | 2.2x per-layer vs the loop; banded prefill launches (m_tile 16/32/64) cover batches whose expert counts fit 256 rows |
| CUDA graphs FULL (sizes 1..24) | ON | capture sizes capped at the fused reach (24 x top-8 = 216 <= 256); `max_num_seqs` must be <= mamba blocks (125) |
| Post-load fused warmup (`process_weights_after_loading`) | ON | creates the ext device context before capture; capture fails without it |
| Capture-safe routing (`scatter_add_`, no `bincount`) | ON | `bincount` is not capturable |
| fp8 KV cache (auto) | ON | SM90 sparse-MLA route; `kv_cache_dtype=auto` resolves to the backend default |
| Chunked prefill + prefix caching | ON | vLLM defaults for this model |
| n-gram spec decode | PARKED | measured in the probe run; acceptance-dependent (see below) |
| DFlash2 spec decode (M4) | NOT PORTED | the big multiplier (2.25x on the sibling lane); next milestone |
| Recon tier (experts > 256 rows, very long prefills) | NOT PORTED | batches above the band limit fall back to the per-expert loop |
| int8 GEMV (M2b) | NOT PORTED | dense-linears compute path; weight-traffic bound at batch 1 |
| async scheduling | AVAILABLE | helps concurrent-serving throughput; no single-stream effect |
| TP=2 / EP=2 | PARKED | after the single-GB10 work, per operator decision |

## Measured (this profile, AR-only — no DFlash2)

| Config | Decode (2k prompt, 256 new) | Prefill |
|---|---|---|
| per-expert loop, eager | 7.67 tok/s | — |
| fused MoE, eager | 15.13 tok/s | — |
| fused MoE + graphs (1..16) | 16.80 tok/s | — |
| **this profile** (graphs 1..24 + banded) | **17.88 tok/s** | **343.2 tok/s** (3608-tok prompt, 10.5 s) |

Sanity in the profile run: `2+2 = 4` chain, coherent. Log:
`work/logs/profile-single-gb10-20260919.log`.

Numbers are from greedy runs on the pack at
`/home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3`; full command + log in
`work/logs/`. Nothing here is a quality claim: Q200v2 / NIAH / BFCL have not
been run on this engine.
