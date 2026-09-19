#!/usr/bin/env bash
# GB10 cluster bootstrap for vllm-exl3-sm121.
# Usage:
#   scripts/bootstrap_cluster.sh --check   # platform + pin check only
#   scripts/bootstrap_cluster.sh           # full bootstrap (long first build)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-bootstrap}"
VENV="${EXL3_VENV:-$HOME/venvs/exl3_sm121}"
WORK="$REPO_ROOT/work"
LOCK="$REPO_ROOT/runtime.lock.json"

lock() { python3 -c "import json,sys; d=json.load(open('$LOCK')); print($1)"; }
VLLM_COMMIT="$(lock "d['vllm']['commit']")"
EXL3_COMMIT="$(lock "d['exllamav3']['commit']")"
ARCH_LIST="$(lock "d['platform']['torch_cuda_arch_list']")"

echo "== platform check =="
ARCH="$(uname -m)"
echo "arch: $ARCH"
[[ "$ARCH" == "aarch64" ]] || echo "WARN: not aarch64; this repo targets GB10 (aarch64)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -q | grep -i "addressing mode" || echo "WARN: cannot read addressing mode"
else
  echo "WARN: nvidia-smi not found"
fi
command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "WARN: nvcc not found"

if [[ "$MODE" == "--check" ]]; then
  echo "== pins =="
  echo "vLLM:      $VLLM_COMMIT"
  echo "exllamav3: $EXL3_COMMIT"
  echo "checked only; no changes made"
  exit 0
fi

echo "== venv =="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
# Editable installs below run --no-build-isolation, so the build backend must
# exist in the venv (fresh venvs ship without setuptools; pip then fails with
# BackendUnavailable: Cannot import 'setuptools.build_meta').
python -m pip install setuptools wheel setuptools_rust
# vLLM's setup.py imports torch at build time; with --no-build-isolation it
# must be pre-installed at the locked version (lock: torch 2.13.0+cu130).
python -m pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130

echo "== vLLM @ pinned commit =="
mkdir -p "$WORK"
if [[ ! -d "$WORK/vllm/.git" ]]; then
  git clone https://github.com/vllm-project/vllm.git "$WORK/vllm"
fi
git -C "$WORK/vllm" fetch origin "$VLLM_COMMIT" --depth 1 || true
git -C "$WORK/vllm" checkout "$VLLM_COMMIT"
python -m pip install --no-build-isolation -e "$WORK/vllm"

echo "== plugin =="
python -m pip install -e "$REPO_ROOT"

echo "== EXL3 extension =="
TORCH_CUDA_ARCH_LIST="$ARCH_LIST" python csrc/setup_ext.py build_ext --inplace

echo "== smoke =="
python -m pytest tests -q
python - <<'EOF'
from vllm_exl3_sm121 import kernels
print("ext probe:", kernels.probe())
EOF

echo "bootstrap complete. Next: serve a pack (docs/BUILD-GB10.md), and record"
echo "receipts (nvidia-smi, first-boot log, placement receipts) in your evidence dir."
