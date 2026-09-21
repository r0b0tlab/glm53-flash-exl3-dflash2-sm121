#!/usr/bin/env bash
# GLM-5.3-Flash EXL3 + DFlash2 K=5 — Q200v2 BFCL v4 multi_turn_base hard-20 lane.
# Fresh BFCL_PROJECT_ROOT + timing sidecar per attempt (no resume for claim runs).
set -u
ROOT=/home/r0b0tdgx/vllm-exl3-sm121
RUNNER=$ROOT/work/q200-runner
EPOCH=${Q200_EPOCH:-$(date -u +%Y%m%dT%H%M%SZ)}
EV=$ROOT/work/q200-evidence/$EPOCH
mkdir -p "$EV"

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export Q200_SERVED_MODEL=glm53-flash-exl3-dflash2
export Q200_BFCL_REGISTRY=glm53exl3-hard20-FC
export Q200_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true,"thinking":true,"reasoning_effort":"low"}'
export Q200_IMAGE_ID=sha256:9ab175696a136534026ebcc7f94d7f3ab5b4bb1970706424054788a85ad26eb0
export Q200_PROFILE_ID=$(sha256sum "$ROOT/work/bench/serve-k5-q200.sh" | cut -d' ' -f1)
export Q200_CANDIDATE_ID=glm53-flash-exl3-2.25hq-tapK3-dflash2k5
export BFCL_PROJECT_ROOT=$EV/bfcl-root-$EPOCH
export Q200_BFCL_TIMING_PATH=$EV/bfcl-timing-$EPOCH.json
export BFCL_NUM_THREADS=1
export BFCL_HTTP_TIMEOUT=1800
export BFCL_MAX_RETRIES=1
export BFCL_MAX_TOKENS=8192

mkdir -p "$BFCL_PROJECT_ROOT"
cd "$RUNNER"

echo "== inspect ==" 
python3 scripts/run_bfcl_hard20.py inspect > "$EV/bfcl-inspect.json" 2>&1 && echo "inspect ok"

echo "== run =="
# inspect imports the BFCL lib, which drops .file_locks/ into BFCL_PROJECT_ROOT;
# the run witness requires a fresh root, so reset it right before the run.
rm -rf "$BFCL_PROJECT_ROOT" && mkdir -p "$BFCL_PROJECT_ROOT"
python3 scripts/run_bfcl_hard20.py run > "$EV/bfcl-run.log" 2>&1
rc_run=$?
echo "BFCL_RUN_RC=$rc_run"

echo "== evaluate =="
python3 scripts/run_bfcl_hard20.py evaluate > "$EV/bfcl-evaluate.log" 2>&1
rc_eval=$?
echo "BFCL_EVAL_RC=$rc_eval"

echo "== status =="
python3 scripts/run_bfcl_hard20.py status > "$EV/bfcl-status.json" 2>&1
tail -20 "$EV/bfcl-status.json"
