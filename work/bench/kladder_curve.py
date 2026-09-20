#!/usr/bin/env python3
"""Concurrency curve probe: C1/C2/C4/C8/C16 aggregates + c1 median (5 reps).

Same prompts/timing convention as kladder.py so numbers are comparable with
the banked K=5 leg (c1 median 30.94, C16 best 39.4). Tag: curve-<label>.
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
TAG = sys.argv[1] if len(sys.argv) > 1 else "curve"

PROMPT = ("Write a detailed technical explanation, about 300 words, "
          "of how a transformer attention head computes its output. "
          "Cover the query, key, and value projections.")
CANARY = "What is 7 plus 5? Reply with only the number."


def post(path, payload, timeout=3600.0):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(prompt, max_tokens, temperature=0.0):
    t0 = time.perf_counter()
    body = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": max_tokens,
    })
    dt = time.perf_counter() - t0
    usage = body.get("usage", {})
    return {"completion_tokens": usage.get("completion_tokens") or 0,
            "finish": body["choices"][0].get("finish_reason"),
            "elapsed": dt}


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as resp:
        text = resp.read().decode()
    out = {}
    for name in ("vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_accepted_tokens_total",
                 "vllm:spec_decode_num_drafts_total"):
        m = re.search(rf"^{name}\s+([0-9.eE+]+)$", text, re.M)
        out[name] = float(m.group(1)) if m else None
    return out


def round_c(c: int, max_tokens: int = 256) -> dict:
    results = []
    lock = threading.Lock()

    def one(i):
        r = chat(f"{PROMPT} (variant {i})", max_tokens)
        with lock:
            results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(c)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.perf_counter() - t0
    total = sum(r["completion_tokens"] for r in results)
    return {"c": c, "seconds": dt, "tokens": total,
            "agg_tok_s": total / dt if dt else 0.0,
            "per_stream": total / dt / c if c else 0.0,
            "finish": sorted({r["finish"] for r in results})}


def main():
    res = {"tag": TAG}
    chat("Say hello.", 32)
    a = chat(CANARY, 256)
    b = chat(CANARY, 256)
    res["canary"] = a
    print(f"[{TAG}] canary finish={a['finish']} {a['completion_tokens']} tok "
          f"in {a['elapsed']:.1f}s", flush=True)

    # c1 median (5 reps, 2048 cap) - comparability with the banked leg
    m0 = metrics()
    speeds = []
    for i in range(5):
        r = chat(PROMPT, 2048)
        speeds.append(r["completion_tokens"] / r["elapsed"])
        print(f"[{TAG}] c1 rep{i}: {r['completion_tokens']} tok in "
              f"{r['elapsed']:.1f}s = {speeds[-1]:.2f} tok/s", flush=True)
    res["c1_speeds"] = speeds
    res["c1_median"] = statistics.median(speeds)
    m1 = metrics()

    # ladder
    ladder = []
    for c in (1, 2, 4, 8, 16):
        r = round_c(c)
        ladder.append(r)
        print(f"[{TAG}] C{c}: {r['tokens']} tok in {r['seconds']:.1f}s = "
              f"{r['agg_tok_s']:.1f} agg ({r['per_stream']:.2f}/stream) "
              f"finish={r['finish']}", flush=True)
    res["ladder"] = ladder
    m2 = metrics()

    def delta(k):
        if m0[k] is None or m2[k] is None:
            return None
        return m2[k] - m0[k]
    drafts, acc, ndrafts = (delta("vllm:spec_decode_num_draft_tokens_total"),
                            delta("vllm:spec_decode_num_accepted_tokens_total"),
                            delta("vllm:spec_decode_num_drafts_total"))
    if ndrafts:
        res["mean_accept_len"] = 1 + acc / ndrafts
        res["draft_accept_pct"] = 100 * acc / drafts if drafts else None
        print(f"[{TAG}] acceptance: AL={res['mean_accept_len']:.2f} "
              f"draft={res['draft_accept_pct']:.1f}%", flush=True)

    with open(f"/tmp/{TAG}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[{TAG}] CURVE_DONE c1_median={res['c1_median']:.2f} "
          f"C16={ladder[-1]['agg_tok_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
