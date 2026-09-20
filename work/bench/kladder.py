#!/usr/bin/env python3
"""K-freeze ladder probe for the DFlash2 serve.

Per K: warmup -> deterministic canary x2 (self-identity) -> c1 decode 2048-out
x5 (median) -> C16 256-out x2 (best) -> acceptance from /metrics deltas.

Usage: kladder.py <tag>   (tag e.g. k5; writes /tmp/kladder-<tag>.json)
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "glm53-flash-exl3-dflash2"
K = sys.argv[2] if len(sys.argv) > 2 else "?"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"

CANARY = "What is 7 plus 5? Reply with only the number."


def post(path: str, payload: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(prompt: str, max_tokens: int, temperature: float = 0.0) -> dict:
    t0 = time.perf_counter()
    body = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    dt = time.perf_counter() - t0
    choice = body["choices"][0]
    usage = body.get("usage", {})
    return {
        "text": choice["message"].get("content"),
        "reasoning_len": len(choice["message"].get("reasoning_content") or ""),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": choice.get("finish_reason"),
        "elapsed": dt,
        "tok_s": (usage.get("completion_tokens") or 0) / dt if dt else 0.0,
    }


def metrics() -> dict:
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as resp:
        text = resp.read().decode()
    out = {}
    for name in ("vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_accepted_tokens_total",
                 "vllm:spec_decode_num_drafts_total",
                 "vllm:kv_cache_usage_perc"):
        m = re.search(rf"^{name}\s+([0-9.eE+]+)$", text, re.M)
        if m:
            out[name] = float(m.group(1))
    return out


def main() -> None:
    res: dict = {"tag": TAG, "k_hint": K}

    # warmup
    chat("Say hello.", 32)

    # canary x2
    c1 = chat(CANARY, 256)
    c2 = chat(CANARY, 256)
    res["canary_text"] = c1["text"]
    res["canary_self_identical"] = c1["text"] == c2["text"]
    print(f"[{TAG}] canary: {c1['text']!r} self-identical={res['canary_self_identical']}", flush=True)

    m0 = metrics()

    PROMPT = ("Write a detailed technical explanation, about 300 words, "
              "of how a transformer attention head computes its output. "
              "Cover the query, key, and value projections.")
    c1_speeds = []
    for i in range(5):
        r = chat(PROMPT, 2048)
        c1_speeds.append(r["tok_s"])
        print(f"[{TAG}] c1 rep{i}: {r['completion_tokens']} tok in {r['elapsed']:.1f}s "
              f"= {r['tok_s']:.2f} tok/s (finish={r['finish']})", flush=True)
    res["c1_tok_s"] = c1_speeds
    res["c1_median"] = statistics.median(c1_speeds)

    m1 = metrics()

    def c16_round(tag: str) -> float:
        results: list[dict] = []
        lock = threading.Lock()

        def one(i: int) -> None:
            r = chat(f"{PROMPT} (variant {i})", 256)
            with lock:
                results.append(r)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=one, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.perf_counter() - t0
        total = sum(r["completion_tokens"] or 0 for r in results)
        agg = total / dt
        print(f"[{TAG}] C16 {tag}: {total} tok in {dt:.1f}s = {agg:.1f} agg tok/s", flush=True)
        return agg

    c16_a = c16_round("a")
    c16_b = c16_round("b")
    res["c16_agg"] = [c16_a, c16_b]
    res["c16_best"] = max(c16_a, c16_b)

    m2 = metrics()
    drafts = m2.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    acc = m2.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    ndrafts = m2.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    if ndrafts:
        res["mean_accept_len"] = 1 + acc / ndrafts
        res["draft_accept_pct"] = 100 * acc / drafts if drafts else None
    print(f"[{TAG}] acceptance: AL={res.get('mean_accept_len')} "
          f"draft%={res.get('draft_accept_pct')}", flush=True)

    with open(f"/tmp/kladder-{TAG}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[{TAG}] KLADDER_DONE c1_median={res['c1_median']:.2f} c16_best={res['c16_best']:.1f}", flush=True)


if __name__ == "__main__":
    main()
