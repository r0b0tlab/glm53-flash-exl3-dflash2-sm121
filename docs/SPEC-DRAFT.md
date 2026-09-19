# Milestone M4 — DFlash2 spec decode in vLLM (survey 2026-09-19)

Base v0.30.0rc1 already carries the machinery; this is a port, not a build.

## What exists upstream (pinned source, `work/vllm`)

- `Glm5NextForConditionalGeneration` registered (`models/glm5next`) — our
  multimodal target arch is present.
- `DFlash2DraftModel` registered -> `models/qwen3_dflash2.py`
  (`DFlash2Qwen3ForCausalLM`, 290 lines, generic over hidden_size/taps,
  built on Qwen3 decoder layers).
- `v1/spec_decode/dflash.py` (321 lines) + `draft_model.py` proposers;
  draft config flows through `speculative_config.draft_model_config`.

## Work items (M4)

1. GLM-flavored DFlash2 draft model: same structure as `qwen3_dflash2.py`
   with Glm5Next decoder layers, hidden 4096, tap points {5,14,24,33,42}
   (mirror the draft checkpoint, not Qwen3's). New file in our plugin or
   an anchored vLLM patch per AGENTS.md rule 9 (prefer plugin).
2. Its linear layers use Exl3LinearMethod (M3) for the 3.0bpw EXL3 draft
   pack; conv projections + selector stay dense (matches quant contract).
3. Serve flag: `--speculative-model <draft> --speculative-config` via the
   existing dflash proposer. Target for parity: exllamav3-measured AL 6.39
   batch-1 (receipt: `projects/exl3-glm53-flash/logs/acceptance-dflash2.json`).
4. Multimodal path rides the registered Glm5Next class; vision tower loads
   dense (vb16 contract). TP=2 via geometry.py (whole-expert EP).

## Non-goals

- Upstreaming the GLM draft model (private fork milestone first).
- Eagle/MTP proposers for this target (DFlash2 only, per campaign spec).
