#!/usr/bin/env bash
# SM12X-LLM-BENCH systems profile — GLM-5.3-Flash EXL3 + DFlash2 K=5 serve.
# Lanes: canary, latency, concurrency, throughput, niah (single 25/50/90 +
# multi-key 33/66 of the served window). BFCL lanes stay PROTOCOL (the Q200v2
# kit carries BFCL hard-20 for this lane).
set -u
cd /home/r0b0tdgx/projects/SM12X-LLM-BENCH
EPOCH=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/home/r0b0tdgx/vllm-exl3-sm121/work/bench/sm12x-systems
mkdir -p "$OUT"
mkdir -p "$OUT/$EPOCH"

.venv/bin/sm12x-bench run \
  --profile systems \
  --no-tmux --display-mode plain \
  --base-url http://127.0.0.1:8000/v1 \
  --model glm53-flash-exl3-dflash2 \
  --output "$OUT/$EPOCH" \
  > "$OUT/$EPOCH/run.log" 2>&1
rc=$?
echo "SM12X_RC=$rc"
python3 - <<EOF
import json, pathlib
p = pathlib.Path("$OUT/$EPOCH/report.json")
if p.exists():
    d = json.loads(p.read_text())
    print(json.dumps({k: d.get(k) for k in ("run_status", "invalid_for_publish", "profile", "model")}, indent=1))
    for lane in d.get("lanes", []):
        print(f"  {lane.get('lane_id'):12s} {lane.get('status'):10s} {str(lane.get('summary'))[:120]}")
else:
    print("no report.json")
EOF
