# Concurrency resolution: per-step host cost on GB10 (measured to the bottom)

Investigation 2026-09-20, K=5 DFlash2 serve, single GB10. Tools: nsys
(launch/start/stop deferred collection), py-spy record/dump, CUPTI sqlite
forensics, primitive microbenches, SM12X-style curve probe.

## Symptom

Aggregate decode plateaus from C2: C1 ≈ 26–31 tok/s (c1 median ≈ 30.6–31.3),
C2 ≈ 37, C4 ≈ 38, C8 ≈ 38, C16 ≈ 38. Per-step cost strictly linear in tokens;
GPU kernel time ≈ 3 ms/step against a ~108 ms C1 step wall.

## Measured platform law (microbench, live serve resident)

| primitive | idle | behind in-flight GPU work |
|---|---|---|
| pageable D2H copy | 17 µs | **~2000 µs** (host blocks at enqueue) |
| pinned D2H copy (non_blocking) | — | 7 µs enqueue, **~2000 µs to completion** (event sync) |
| pinned H2D copy | — | 7 µs |
| D2D copy | — | 8 µs |
| event record + sync (no data) | 5 µs | 5 µs |

⇒ **On this platform a D2H *completion* costs ≈ 2 ms whenever GPU work is in
flight** (ATS page migration / table walk), while H2D/D2D/event syncs are free.

## Where the step time goes (live py-spy record, 45 s, engine main thread)

- attention metadata build ≈ 20 % — FlashInfer sparse plan + per-seq mamba
  block tables + kpool tail slot mapping + `_kv_lens_host`
- split-phase output wait (`get_output → copy_event.synchronize`) ≈ 21 % —
  waiting on the per-step D2H completions (vLLM v1's output readback)
- drafter `precompute_and_store_context_kv` inside `propose` ≈ 10 %
- sampling/verify/input-prep/reporting ≈ the rest
- total host ≈ 100 ms/step at C1, growing ~26 ms per extra seq (C16 step ≈ 1.4 s)

## Bugs found and fixed en route (kept)

1. **MoE capture path (real bug, fixed @ 22ac41f).** `a > rows` under capture
   fell to the per-expert loop (`torch.unique` — capture-unsupported) → any
   capture size above ~32 tokens failed the boot. Now dispatches the all-fused
   launch under capture; capture sizes to 96 tokens (16 lanes × (1+K)) boot.
2. **Per-step D2H in the sparse-MLA metadata (`_kv_lens_host`, patch-0008
   file).** The sync-free host-lens path is gated on `not async_scheduling`;
   with async scheduling (default) every step did `positions.cpu()` — one D2H
   per step. Removed via `--no-async-scheduling` (the flag stays; the other
   host costs dominate, so the net curve is unchanged).

## Flags A/B'd against the curve (all neutral)

- capture sizes [1..96] (fixed the boot bug; graphs were not the limiter)
- `--no-async-scheduling` (removes the kv_lens D2H; curve unchanged)
- `--max-num-batched-tokens 2048` (the MiaAI recipe default; unchanged)

## Class comparison (same model, GB10, vLLM + DFlash2)

- Sibling lane (2×GB10, 0.28 overlay): C1 68.4 / C2 124 / C4 214.5 / C6 282.8
  agg (think-off, K8, code-heavy). Consistent with the same host-step model:
  ~7 tokens/step × ~100 ms steps at C1, ~6×7 tokens/step at C6.
- MiaAI (2×GB10, EXL3, K7): ladder ×1 35 / ×2 50 / ×4 67 / ×8 8.5–29 / ×16 6.8;
  "the interactive knee is ~4 lanes"; speed "set by the drafter's acceptance
  (code 35–44 solo, prose ~18)".
- This engine, high-AL workload (repetitive, AL 6.6–7.4): 50.2 tok/s C1 —
  in-class. Prose AL ≈ 2.65 → C1 ≈ 17–24 and the C2+ plateau at ≈ 38.

## Conclusion

The plateau is the vLLM v1 host step on this platform (one ~100 ms host step,
~2 ms per D2H completion, per-seq metadata growth) — not an EXL3/DFlash2
defect, and not config-tunable with the flags tested. Effective throughput is
(tokens per step) ÷ (host step) — so the levers that matter are acceptance per
step (workload/prompt class) and upstream-level host-path work (cache the
metadata builds; batch the per-seq loops; avoid D2H completions in the hot
path). Residual local levers if pursued: prefix-caching-off / mamba cache-mode
experiments (trade cache hits for host time; not recommended), and
contribution-class upstream optimizations.

Evidence: `work/logs/serve-k5-nsys.log`, `/tmp/report1.nsys-rep` (C1 trace),
`/tmp/eng-profile.json` (py-spy speedscope), curve logs
`work/logs/curve-{k5fix,nosync,mntb}.log`, microbench outputs in this file.
