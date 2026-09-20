#!/usr/bin/env python3
"""Class-convention perf bench: structured / code / prose x C1 median(5) + ladder.

Mirrors the published EXL3/DFlash2 lanes (MiaAI: code 35-44 solo, prose ~18;
single-Spark: structured 64, prose 25, C4 agg 182) so numbers are comparable.
Tag: bench-class-<label>.  chat kwargs: reasoning_effort=low (lightest this
template has; the class rows are think-off).
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
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
EFFORT = sys.argv[2] if len(sys.argv) > 2 else "low"

PROMPTS = {
    "structured": (
        "Output a JSON array of 25 countries. For each, an object with keys "
        "\"country\", \"capital\", \"population_millions\" (number, 1 decimal). "
        "Output only the JSON, no markdown fences, no commentary."
    ),
    "code": (
        "Write a complete Python class RingBuffer with methods push, pop, "
        "is_empty, and __len__, backed by a fixed-size list with head/tail "
        "indices. Include type hints and a short docstring per method. "
        "Output only the code."
    ),
    "prose": (
        "Write a short story, about 200 words, about a lighthouse keeper "
        "who finds a message in a bottle. Use varied prose."
    ),
}


def post(path, payload, timeout=3600.0):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(prompt, max_tokens, salt=None):
    content = prompt if salt is None else f"{prompt}\n\n<!-- {salt} -->"
    t0 = time.perf_counter()
    body = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"reasoning_effort": EFFORT},
    })
    dt = time.perf_counter() - t0
    u = body.get("usage", {})
    m = body["choices"][0]["message"]
    return {
        "tok": u.get("completion_tokens") or 0,
        "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
        "elapsed": dt, "finish": body["choices"][0].get("finish_reason"),
        "text": (m.get("content") or "")[:60],
    }


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as resp:
        text = resp.read().decode()
    out = {}
    for name in ("vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_accepted_tokens_total",
                 "vllm:spec_decode_num_drafts_total",
                 "vllm:spec_decode_num_accepted_tokens_per_pos_sum"):
        m = re.search(rf"^{name}\s+([0-9.eE+]+)$", text, re.M)
        out[name] = float(m.group(1)) if m else None
    return out


def ladder_round(c, prompt, max_tokens=256):
    res = []
    lock = threading.Lock()

    def one(i):
        r = chat(prompt, max_tokens, salt=f"v{i}")
        with lock:
            res.append(r)

    t0 = time.perf_counter()
    ts = [threading.Thread(target=one, args=(i,)) for i in range(c)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    total = sum(r["tok"] for r in res)
    return {"c": c, "secs": dt, "tok": total, "agg": total / dt if dt else 0}


def main():
    out = {"tag": TAG, "effort": EFFORT}
    chat("Say hello.", 32)
    m0 = metrics()

    # per-class C1 median (5 reps, 2048 cap, unique salt to defeat prefix cache)
    for name, prompt in PROMPTS.items():
        speeds = []
        for i in range(5):
            r = chat(prompt, 2048, salt=f"r{i}")
            speeds.append(r["tok"] / r["elapsed"])
            print(f"[{TAG}] {name} rep{i}: {r['tok']} tok in {r['elapsed']:.1f}s "
                  f"= {speeds[-1]:.2f} tok/s (reason={r['reasoning']} "
                  f"finish={r['finish']})", flush=True)
        out[f"{name}_speeds"] = speeds
        out[f"{name}_median"] = statistics.median(speeds)
        print(f"[{TAG}] {name} MEDIAN: {out[f'{name}_median']:.2f} tok/s", flush=True)
    m1 = metrics()

    # ladder on structured (lanes convention)
    lad = []
    for c in (1, 2, 4, 6, 8, 16):
        r = ladder_round(c, PROMPTS["structured"])
        lad.append(r)
        print(f"[{TAG}] C{c}: {r['tok']} tok in {r['secs']:.1f}s = "
              f"{r['agg']:.1f} agg", flush=True)
    out["ladder"] = lad
    m2 = metrics()

    def d(k):
        return None if (m0[k] is None or m2[k] is None) else m2[k] - m0[k]
    drafts, acc, ndrafts = (d("vllm:spec_decode_num_draft_tokens_total"),
                            d("vllm:spec_decode_num_accepted_tokens_total"),
                            d("vllm:spec_decode_num_drafts_total"))
    if ndrafts:
        out["AL"] = 1 + acc / ndrafts
        out["draft_accept_pct"] = 100 * acc / drafts if drafts else None
        print(f"[{TAG}] acceptance AL={out['AL']:.2f} "
              f"draft={out['draft_accept_pct']:.1f}%", flush=True)

    with open(f"/tmp/{TAG}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[{TAG}] CLASS_DONE", flush=True)


if __name__ == "__main__":
    main()
