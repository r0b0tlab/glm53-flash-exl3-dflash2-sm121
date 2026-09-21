#!/usr/bin/env bash
# GLM-5.3-Flash EXL3 + DFlash2 K=5 — Q200v2 text-180 lane (single GB10).
set -u
ROOT=/home/r0b0tdgx/vllm-exl3-sm121
RUNNER=$ROOT/work/q200-runner
EPOCH=${Q200_EPOCH:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ID="glm53-exl3-dflash2-k5-q200v2-$EPOCH"
EV=$ROOT/work/q200-evidence/$RUN_ID
mkdir -p "$EV"
cd "$RUNNER"

python3 scripts/run_quality_set.py \
  --base-url http://127.0.0.1:8000 \
  --run-id "$RUN_ID" \
  --model glm53-flash-exl3-dflash2 \
  --image-id sha256:9ab175696a136534026ebcc7f94d7f3ab5b4bb1970706424054788a85ad26eb0 \
  --profile-id glm53exl3-dflash2-k5-serve32768 \
  --candidate-id glm53-flash-exl3-2.25hq-tapK3-dflash2k5 \
  --workers 4 --max-tokens 16384 --timeout 1800 \
  > "$EV/quality-run.log" 2>&1
rc=$?
echo "QUALITY_RC=$rc"
cp "$RUNNER/$RUN_ID.summary.json" "$EV/" 2>/dev/null || true
cp "$RUNNER/$RUN_ID.rows.jsonl" "$EV/" 2>/dev/null || true
ls -la "$EV" | tail -5
