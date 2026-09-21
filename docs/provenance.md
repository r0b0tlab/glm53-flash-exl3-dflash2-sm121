# provenance

Per-file record of where code in this repository comes from. Update in the
same commit as any change.

## Our code (MIT)

Everything under `src/`, `tests/`, `scripts/`, `docker/`, `csrc/`,
`patches/`, `docs/` unless listed below.

## Vendored (MIT, turboderp-org/exllamav3 @ v1.5.0 / 0740edc2)

`vendored/exllamav3/ext/**` — copied verbatim, unmodified. See
`vendored/VENDOR.md` for the file list and the reproduction script.

## Design references (no code copied)

| Source | License | What was learned |
|---|---|---|
| vcruz305/vllm-exl3 | AGPL-3.0 | Feature surface for parity: mixed-K K1-K8, TP/EP geometry resolution, UVA guards, embedding/n-gram method, prescan. Behavior only. |
| vcruz305/exllamav3 (feat/gb10-ats-load) | MIT | ATS placement split semantics (copy-except-pattern), HUGEPAGE advice, DSpark integration (v4.1 milestone). MIT-licensed; may be vendored later with per-file provenance when the v4.1 milestone starts. |
| dphnAI/sonar | AGPL-3.0 | EXL3 config contract shape, 16-aligned shard rule, min_capability 80, correctness-first loop pattern. Behavior only. |
| vcruz305 DSV4.1 recipes | AGPL-3.0 | Gate design (min-fit ladder, OOM guards, receipts, runtime.lock pattern). Behavior only. |
| Reederey87 glm53 kit | (community) | Serving-hardening findings (prefix-cache geometry, fairness caps, spinwait). Behavior only. |
