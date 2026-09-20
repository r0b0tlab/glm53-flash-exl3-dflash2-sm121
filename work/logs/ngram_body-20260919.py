"""Single-GB10 optimize profile — every lever, measured.

Levers in this run:
- fused MoE kernel (exl3_moe + gather, deterministic slots) with banded
  prefill launches
- CUDA graphs FULL, capture sizes [1,2,4,8,16,24]
- max_num_seqs=64 (hybrid mamba block budget)
- gpu_memory_utilization=0.85, fp8 KV (auto), chunked prefill
"""
import time

from vllm import LLM, SamplingParams

PACK = '/home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3'


def main() -> None:
    llm = LLM(model=PACK, quantization='exl3', tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=4096,
              max_num_seqs=24, trust_remote_code=True, enforce_eager=False,
              compilation_config={
                  "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24]},
              speculative_config={
                  "method": "ngram", "num_speculative_tokens": 5,
                  "prompt_lookup_max": 5, "prompt_lookup_min": 2})
    print('LOAD_OK', flush=True)

    out = llm.generate(['What is 2+2? Answer with the number.'],
                       SamplingParams(max_tokens=64, temperature=0.0))[0]
    print('SANITY:', repr(out.outputs[0].text[:120]), flush=True)

    filler = ('The quick brown fox jumps over the lazy dog. ' * 45)
    perf_prompt = filler + "\nSummarize the sentence above in five words."
    t0 = time.time()
    out = llm.generate([perf_prompt],
                       SamplingParams(max_tokens=256, temperature=0.0))[0]
    dt = time.time() - t0
    n = len(out.outputs[0].token_ids)
    print(f'DECODE: {n} tok in {dt:.2f}s = {n / dt:.2f} tok/s', flush=True)

    story = ("Write a short story, about 200 words, about a lighthouse "
             "keeper who finds a message in a bottle. Use varied prose.")
    t0 = time.time()
    out = llm.generate([story],
                       SamplingParams(max_tokens=256, temperature=0.0))[0]
    dt = time.time() - t0
    n = len(out.outputs[0].token_ids)
    print(f'DECODE_STORY: {n} tok in {dt:.2f}s = {n / dt:.2f} tok/s', flush=True)

    big = ('Alpha bravo charlie delta echo foxtrot golf hotel india juliet '
           'kilo lima. ' * 180)
    big_prompt = big + "\nReply with the single word OK."
    t0 = time.time()
    out = llm.generate([big_prompt],
                       SamplingParams(max_tokens=1, temperature=0.0))[0]
    dt = time.time() - t0
    nprompt = len(out.prompt_token_ids)
    print(f'PREFILL: {nprompt} prompt tok in {dt:.2f}s = '
          f'{nprompt / dt:.1f} tok/s', flush=True)
    print('PROFILE_DONE', flush=True)


if __name__ == '__main__':
    main()
