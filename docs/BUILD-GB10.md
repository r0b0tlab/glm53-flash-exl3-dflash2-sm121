# Building on the GB10 cluster

Prerequisites per node: DGX Spark (GB10, aarch64, sm_121, CUDA 13.0), ATS
addressing mode, >= 450 GiB free NVMe, fast local network between nodes for TP.

## 1. Check the platform

```bash
uname -m                                   # aarch64
nvidia-smi -q | grep -i "addressing mode"  # ATS
nvcc --version | tail -1                   # CUDA 13.x
```

## 2. Bootstrap (one node first)

```bash
git clone <this repo> && cd glm53-flash-exl3-dflash2-sm121
bash scripts/bootstrap_cluster.sh --check   # dry run: verify pins and tools
bash scripts/bootstrap_cluster.sh           # clone pinned vLLM, install plugin, build ext
```

The script:

1. creates/completes a venv (`~/venvs/exl3_sm121`),
2. clones vLLM at the pinned commit into `work/vllm` and installs it
   (`--no-build-isolation`),
3. installs this plugin (`pip install -e .`),
4. builds the EXL3 extension from `vendored/` + `csrc/` with
   `TORCH_CUDA_ARCH_LIST="12.0;12.1"`,
5. runs the smoke checks (`pytest tests`, extension probe, pack validation on
   a real pack if present).

First build is long (vLLM ARM64 build). Do it inside tmux.

## 3. Serve a pack (once the extension milestone lands)

```bash
# inside the venv, from the repo root
python -m vllm.entrypoints.openai.api_server \
  --model /models/qwen38-27b-exl3 \
  --quantization exl3 \
  --gpu-memory-utilization 0.86 \
  --max-model-len <ctx>
```

Serve flags per model live in the per-model recipe repos (house pattern). The
placement plan for each host comes from `vllm_exl3_sm121.placement.decide(...)`
receipts in the serve log.

## 4. Multi-node (TP=2)

Use the fleet playbook: direct QSFP/ConnectX-7 link, `NCCL_NET=IB`, GID
pinning, one process per node, worker first. The fork does not change NCCL
behavior; reuse the profiles from the existing SM121 packages.
