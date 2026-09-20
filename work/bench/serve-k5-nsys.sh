#!/bin/bash
# Serve the winning profile under `nsys launch` (deferred collection: no
# tracing until `nsys start`), so a live C-burst can be kernel-profiled.
set -u
LOG=/home/r0b0tdgx/vllm-exl3-sm121/work/logs/serve-k5-nsys.log
source /home/r0b0tdgx/venvs/exl3_sm121/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VLLM_LOGGING_LEVEL=INFO
export EXL3_K=${EXL3_K:-5}
export PATH=/usr/local/cuda/bin:$PATH
export TMPDIR=/home/r0b0tdgx/.cache/nsys-tmp
mkdir -p "$TMPDIR"
sudo -n sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches' || true

UNIT=exl3-nsys-$(date +%H%M%S)
systemd-run --user --scope -p MemoryMax=112G -p MemorySwapMax=6G \
  --unit="$UNIT" -- \
  nsys launch --session-new glm53prof -- \
  vllm serve /home/r0b0tdgx/models/glm-5.3-flash/exl3-2.25hq-tapK3 \
    --quantization exl3 \
    --trust-remote-code \
    --served-model-name glm53-flash-exl3-dflash2 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/home/r0b0tdgx/models/glm-5.3-flash-dflash2/exl3-3.00bpw\",\"num_speculative_tokens\":$EXL3_K}" \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,36,48,60,72,84,96]}' \
    --reasoning-parser glm47 \
    --host 127.0.0.1 --port 8000 \
  > "$LOG" 2>&1
echo "SERVE_EXIT=$?"
